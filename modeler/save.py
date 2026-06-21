from saver import *  # noqa: F401,F403


if __name__ == "__main__":
    artifacts = train_full_traffic_pipeline(DATA_PATH, run_cv=True)
    print("Training complete")
    print(artifacts["training_summary"])
    print(artifacts["duration_training_summary"])
    save_cascade_artifacts(artifacts, ARTIFACT_DIR)
