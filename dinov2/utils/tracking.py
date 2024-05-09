import os
import json
import datetime
import tarfile
import uuid
import boto3
from botocore.exceptions import NoCredentialsError


class ExperimentTracker:
    def __init__(self, output_dir, config):
        self.experiment_name = config['train']['experiment_name']
        self.config = config
        self.runs_dir = os.path.join(output_dir, self.experiment_name)
        self.run_id = str(uuid.uuid4())
        self.run_dir = os.path.join(self.runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        if 'weights_path' in self.config['train'].keys():
            self.weights_path = os.path.join(self.run_dir, self.config['train']['weights_path'])
            os.makedirs(self.weights_path, exist_ok=True)
        self.s3_bucket = self.config['train']['s3_bucket']

    def create_metafile(self, tracked_metrics, train_dataset, val_dataset):
        meta = {
            "run_id": self.run_id,
            "model_config": self.config['model'] if 'model' in self.config.keys() else None,
            "train_config": self.config['train'],
            "train_dataset": {"wsis": train_dataset.wsis,
                              "num_wsis": len(train_dataset.wsis),
                              "num_images": len(train_dataset)},
            "val_dataset": {"wsis": val_dataset.wsis if val_dataset else None,
                            "num_wsis": len(val_dataset.wsis) if val_dataset else None,
                            "num_images": len(val_dataset) if val_dataset else None},
            "tracked_metrics": tracked_metrics if tracked_metrics else None,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        metafile_path = os.path.join(self.run_dir, "meta.json")
        with open(metafile_path, 'w') as f:
            json.dump(meta, f)

    def save_run(self, tracked_metrics, train_dataset, val_dataset=None):
        self.create_metafile(tracked_metrics, train_dataset, val_dataset)
        artifact_paths = [self.run_dir]
        artifacts_tar = os.path.join(self.runs_dir, f"{self.run_id}.tgz")
        self.archive_run_artifacts(artifact_paths, artifacts_tar)
        print(f"Train artifacts saved to {artifacts_tar}")
        self.upload_to_s3(artifacts_tar)

    @staticmethod
    def archive_run_artifacts(artifact_paths, output_path):
        with tarfile.open(output_path, "w:gz") as tar:
            for path in artifact_paths:
                tar.add(path, arcname=os.path.basename(path))

    def upload_to_s3(self, tarfile_path):
        s3_client = boto3.client('s3')
        s3_folder = os.path.join('runs', self.experiment_name)
        try:
            s3_key = os.path.join(s3_folder, os.path.basename(tarfile_path))
            s3_client.upload_file(tarfile_path, self.s3_bucket, s3_key)
            print(f"Train artifacts uploaded successfully to s3://{self.s3_bucket}/{s3_key}")
        except FileNotFoundError:
            print("Train artifacts not found")
        except NoCredentialsError:
            print(f"AWS credentials not available so train artifacts not uploaded to s3. "
                  f"Weights stored in {tarfile_path}")