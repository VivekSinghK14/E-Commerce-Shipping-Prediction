# BPMN and UML architecture diagrams

These diagrams are Mermaid source and render in GitHub, supported VS Code
previews, and Mermaid Live. They describe the complete executable lifecycle,
including the audited management override that deploys the XGBoost prototype
although its evaluation gates failed, and the Hugging Face live demo.

## BPMN-aligned integrated process

```mermaid
flowchart LR
  subgraph DM[Data Management & Analysis]
    A((Business understanding complete)) --> B[Identify data source and requirements]
    B --> C[Acquire raw CSV]
    C --> C2[Remove Returned orders: keep Delivered and Delayed]
    C2 --> D[Describe data]
    D --> E[Explore and assess quality]
    E --> F{Data sufficient and contract valid?}
    F -- No --> B
    F -- Yes --> G[Prepare and engineer order-time features]
    G --> H[Develop and validate data pipeline]
    H --> FS[(Feature store)]
  end

  subgraph AM[Analytical Modeling]
    FS --> I{Hypothesis needed?}
    I -- Yes --> J[Define and record hypotheses]
    I -- No --> K[Select candidate algorithms]
    J --> K
    K --> L[Define model test design]
    L --> M[Develop candidate models]
    M --> MS[(MLflow metadata store)]
    M --> MR[(Candidate model registry)]
    M --> N[Assess validation performance and business factors]
    N --> O{Performance sufficient?}
    O -- No --> P[Further experiments or data modification]
    P --> K
    O -- Yes --> Q[Integrated analytical pipeline: XGBoost selected]
  end

  subgraph EV[Domain Expertise / Strategy & Project Management]
    Q --> R[Assess untouched-test results]
    R --> S[Cost-benefit and business-objective assessment]
    S --> T{Business goals satisfied?}
    T -- Yes --> V[Process evaluation and checkpoint]
    T -- No --> OV{Management override recorded?}
    OV -- No --> U((Return to business understanding))
    OV -- Yes --> OVR[Record actor, rationale, failed gates in audit log]
    V --> W{Proceed to deployment?}
    W -- No --> U
    W -- Yes --> X[Impact assessment]
    OVR --> X
  end

  subgraph DP[Data Infrastructure & Operations - Deployment]
    X --> Y[Plan deployment, monitoring and maintenance]
    Y --> Z[Deployment tests and smoke test]
    Z --> AA{Tests successful?}
    AA -- No --> Y
    AA -- Yes --> AB[Business integration: FastAPI service and Hugging Face Space]
    AB --> AC((Technical system available))
  end

  subgraph UT[Utilization and Continuous Training]
    AC --> AD[Domain-user utilization]
    AD --> AE[Monitoring, outcome feedback and support tickets]
    AE --> AF{Maintenance or retraining needed?}
    AF -- Yes --> AG[Versioned guarded retraining]
    AG --> AE
    AF -- No --> AH{Continue system?}
    AH -- Yes --> AD
    AH -- No --> AI((System discontinuation))
  end
```

## UML component diagram

```mermaid
classDiagram
  class PipelineConfig {
    +Path raw_data
    +Path artifacts
    +business thresholds
    +economic assumptions
  }
  class DataStage {
    +collect_and_prepare(config)
    +remove Returned orders
    +validate input/output contracts
  }
  class FeatureStore {
    +X_train / X_validation / X_test
    +targets
    +scaler.joblib
    +feature_manifest.json
  }
  class ModelStage {
    +develop_and_assess_models(config)
    +leakage-safe candidate comparison
    +business factors per model
  }
  class MLflowStore {
    +candidate runs
    +parameters and metrics
    +models
    +evaluation run
  }
  class EvaluationStage {
    +evaluate_and_checkpoint(config)
    +cost-benefit analysis
    +business objective assessment
  }
  class ModelRegistry {
    +candidate_bundle.joblib
    +candidate_manifest.json
    +champion_bundle.joblib
    +champion_manifest.json
  }
  class LifecycleStage {
    +approve_deployment_override(actor, rationale)
    +manage_system_lifecycle(discontinue/resume)
    +lifecycle_decision_log.jsonl
  }
  class CICDComponent {
    +install dependencies
    +run all tests
  }
  class ServingComponent {
    +FastAPI endpoints
    +raw/processed/batch prediction
    +HTTP 503 without approval or override
    +override warning on every response
  }
  class OperationsStore {
    +SQLite predictions
    +outcome feedback
    +support tickets
    +utilization summary
  }
  class MonitoringComponent {
    +health and pipeline status
    +prediction/error counters
    +Prometheus metrics
    +prediction event log
  }
  class BrowserExport {
    +export_browser_model.py
    +model.json + predictor.js
  }
  class HuggingFaceSpace {
    +static live demo
    +in-browser XGBoost inference
  }
  class RetrainingPipeline {
    +detect triggers()
    +run versioned pipeline()
    +promote only if approved()
  }

  PipelineConfig --> DataStage
  DataStage --> FeatureStore
  FeatureStore --> ModelStage
  ModelStage --> MLflowStore
  ModelStage --> ModelRegistry : candidate
  ModelStage --> EvaluationStage
  EvaluationStage --> MLflowStore
  EvaluationStage --> ModelRegistry : promote if all gates pass
  LifecycleStage --> ModelRegistry : champion by override
  CICDComponent --> ServingComponent : tests gate release
  ModelRegistry --> ServingComponent : champion bundle
  ServingComponent --> OperationsStore
  ServingComponent --> MonitoringComponent
  ModelRegistry --> BrowserExport : champion bundle
  BrowserExport --> HuggingFaceSpace : upload_folder
  MonitoringComponent --> RetrainingPipeline : drift/performance trigger
  RetrainingPipeline --> DataStage
  RetrainingPipeline --> ModelStage
  RetrainingPipeline --> EvaluationStage
```

## UML deployment-by-override sequence

```mermaid
sequenceDiagram
  actor Owner as Project owner
  participant OV as deploy_with_override.py
  participant LC as Lifecycle stage
  participant REG as Model registry
  participant API as FastAPI serving
  participant EX as export_browser_model.py
  participant HF as Hugging Face Space

  Owner->>OV: --actor and --reason
  OV->>LC: approve_deployment_override(actor, rationale)
  LC->>REG: verify candidate bundle SHA-256 matches evaluation
  LC->>REG: copy to champion_bundle.joblib, status approved_by_override
  LC->>LC: write deployment_override.json and audit log entry
  API->>REG: load champion bundle at startup
  API-->>Owner: predictions with DEPLOYED_BY_MANAGEMENT_OVERRIDE warning
  Owner->>EX: export champion for the browser
  EX->>REG: read champion trees and threshold
  EX-->>HF: model.json, predictor.js, index.html
  HF-->>Owner: live demo predicts in the browser
```

## UML retraining sequence

```mermaid
sequenceDiagram
  actor Scheduler as Scheduler/Operator
  participant RT as Retraining controller
  participant DS as Data stage
  participant MS as Model stage
  participant MF as MLflow
  participant EV as Evaluation gate
  participant REG as Model registry

  Scheduler->>RT: schedule, new-data, performance, or --force trigger
  RT->>RT: compare source hash, age, observed metric
  alt trigger exists
    RT->>DS: validate contracts and rebuild versioned feature store
    DS-->>RT: quality-approved features
    RT->>MS: train and compare leakage-safe candidates
    MS->>MF: log candidate runs, models, metrics
    MS-->>EV: frozen candidate and threshold
    EV->>EV: untouched-test and business-cost assessment
    EV->>MF: log evaluation and reports
    alt every business gate passes
      EV->>REG: promote candidate to champion
    else any gate fails
      EV-->>RT: reject and preserve existing champion
    end
  else no trigger
    RT-->>Scheduler: skip and report reason
  end
```
