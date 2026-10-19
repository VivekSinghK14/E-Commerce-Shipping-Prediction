// In-browser re-implementation of deployment/app.py prediction:
// order-time feature engineering -> train-fitted scaling -> XGBoost trees.
// Values are cast to float32 exactly as XGBoost does internally.
(function (root) {
  "use strict";
  const f32 = Math.fround;

  function categoryCode(model, column, value) {
    const wanted = String(value).trim().split(/\s+/).join(" ").toLowerCase();
    for (const [label, code] of Object.entries(model.category_maps[column])) {
      if (label.toLowerCase() === wanted) return code;
    }
    throw new Error(`${column} must be one of: ${Object.keys(model.category_maps[column]).join(", ")}`);
  }

  function orderCalendar(isoDate) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(isoDate));
    if (!match) throw new Error("Order_Date must use the format YYYY-MM-DD");
    const [year, month, day] = match.slice(1).map(Number);
    const date = new Date(Date.UTC(year, month - 1, day));
    if (date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day) {
      throw new Error("Order_Date is not a valid calendar date");
    }
    // pandas dayofweek: Monday=0 ... Sunday=6
    return { month, dayOfWeek: (date.getUTCDay() + 6) % 7 };
  }

  function buildFeatures(model, record) {
    const cost = Number(record.Shipping_Cost);
    if (!Number.isFinite(cost) || cost < 0) throw new Error("Shipping_Cost must be a non-negative number");
    const { month, dayOfWeek } = orderCalendar(record.Order_Date);
    const raw = {
      Customer_Region: categoryCode(model, "Customer_Region", record.Customer_Region),
      Product_Category: categoryCode(model, "Product_Category", record.Product_Category),
      Shipping_Mode: categoryCode(model, "Shipping_Mode", record.Shipping_Mode),
      Shipping_Cost: cost,
      Order_Month: month,
      Order_DayOfWeek: dayOfWeek,
    };
    model.numeric_columns.forEach((column, i) => {
      raw[column] = (raw[column] - model.scaler_mean[i]) / model.scaler_scale[i];
    });
    return model.feature_columns.map((column) => f32(raw[column]));
  }

  function deliveredProbability(model, features) {
    let margin = Math.log(model.base_score / (1 - model.base_score));
    for (const [left, right, featureIndex, condition] of model.trees) {
      let node = 0;
      while (left[node] !== -1) {
        node = features[featureIndex[node]] < f32(condition[node]) ? left[node] : right[node];
      }
      margin += condition[node]; // leaf value
    }
    return 1 / (1 + Math.exp(-margin));
  }

  function predict(model, record) {
    const probability = deliveredProbability(model, buildFeatures(model, record));
    const prediction = probability >= model.threshold ? 1 : 0;
    return {
      prediction,
      prediction_label: prediction === 1 ? "Delivered" : "Delayed",
      on_time_probability: probability,
      threshold: model.threshold,
    };
  }

  const api = { buildFeatures, deliveredProbability, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ShippingPredictor = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
