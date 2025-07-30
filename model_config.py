import os

class ModelConfig:
    def __init__(self, model_type="nn", model_dir="model"):
        self.model_dir = model_dir
        self.model_type = model_type.lower()

        if self.model_type == "cnn":
            self.model_name = "best_cnn_model.keras"
        else:
            self.model_name = "best_nn_model.keras"

        self.full_path = os.path.join(self.model_dir, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def get_model_path(self):
        return self.full_path

    def model_exists(self):
        return os.path.exists(self.full_path)
    
    def get_model_name(self):
        return self.model_name
