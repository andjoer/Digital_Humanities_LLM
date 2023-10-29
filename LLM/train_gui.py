import sys
from dataclasses import dataclass, field
from typing import Optional
from PyQt5.QtWidgets import QApplication, QWidget, QFormLayout, QLineEdit, QCheckBox, QComboBox, QVBoxLayout, QPushButton, QTabWidget, QLabel, QFileDialog, QHBoxLayout

from dataclasses import dataclass

from finetune import load_and_train
@dataclass
class ScriptArgs:
    # TrainingConfig fields
    local_rank: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    weight_decay: float
    lora_alpha: int
    lora_dropout: float
    lora_r: int
    max_seq_length: int
    num_train_epochs: int
    optim: str
    lr_scheduler_type: str
    max_steps: int
    eval_steps: int
    warmup_ratio: float
    group_by_length: bool
    save_steps: int
    logging_steps: int
    output_dir: str
    checkpoint: str
    train_eval_dir: str
    merge_and_push: bool
    
    # ModelConfig fields
    model_name: str
    use_4bit: bool
    use_8bit: bool
    use_lora: bool
    use_nested_quant: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    fp16: bool
    bf16: bool
    packing: bool
    gradient_checkpointing: bool
    wnb_project: str
    eval_collator: str
    train_collator: str


app = QApplication(sys.argv)

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create layout and widgets
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Add tabs
        self.tabs.addTab(self.trainingSettingsUI(), "Training Settings")
        self.tabs.addTab(self.modelSettingsUI(), "Model Settings")

        layout.addWidget(self.tabs)

        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.startTraining)
        layout.addWidget(self.start_training_btn)

        self.setLayout(layout)
        self.setWindowTitle('Training Configuration')
        self.show()

    def browseForDirectory(self):
        """Open a file dialog and return the selected directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        return directory
    
    def trainingSettingsUI(self):
        widget = QWidget()
        layout = QFormLayout()

        # Create and add widgets for training settings
        self.local_rank = QLineEdit("-1")
        layout.addRow(QLabel("Local Rank:"), self.local_rank)
        
        self.per_device_train_batch_size = QLineEdit("1")
        layout.addRow(QLabel("Per Device Train Batch Size:"), self.per_device_train_batch_size)
        
        self.per_device_eval_batch_size = QLineEdit("1")
        layout.addRow(QLabel("Per Device Eval Batch Size:"), self.per_device_eval_batch_size)
        
        self.gradient_accumulation_steps = QLineEdit("4")
        layout.addRow(QLabel("Gradient Accumulation Steps:"), self.gradient_accumulation_steps)
        
        self.learning_rate = QLineEdit("2e-4")
        layout.addRow(QLabel("Learning Rate:"), self.learning_rate)
        
        self.max_grad_norm = QLineEdit("0.3")
        layout.addRow(QLabel("Max Grad Norm:"), self.max_grad_norm)
        
        self.weight_decay = QLineEdit("0.001")
        layout.addRow(QLabel("Weight Decay:"), self.weight_decay)
        
        self.lora_alpha = QLineEdit("64")
        layout.addRow(QLabel("LoRA Alpha:"), self.lora_alpha)
        
        self.lora_dropout = QLineEdit("0.1")
        layout.addRow(QLabel("LoRA Dropout:"), self.lora_dropout)
        
        self.lora_r = QLineEdit("512")
        layout.addRow(QLabel("LoRA R:"), self.lora_r)
        
        self.max_seq_length = QLineEdit("4096")
        layout.addRow(QLabel("Max Sequence Length:"), self.max_seq_length)

        self.num_train_epochs = QLineEdit("6")
        layout.addRow(QLabel("Number of Training Epochs:"), self.num_train_epochs)

        self.optim = QComboBox()
        self.optim.addItems(["paged_adamw_32bit", "adam", "sgd"])  # Add as many optimizers as needed
        layout.addRow(QLabel("Optimizer:"), self.optim)

        self.lr_scheduler_type = QComboBox()
        self.lr_scheduler_type.addItems(["constant", "cosine", "linear"])  # Add as many scheduler types as needed
        layout.addRow(QLabel("Learning rate Scheduler:"), self.lr_scheduler_type)

        self.max_steps = QLineEdit("10000")
        layout.addRow(QLabel("Max Steps:"), self.max_steps)

        self.warmup_ratio = QLineEdit("0.01")
        layout.addRow(QLabel("Warmup Ratio:"), self.warmup_ratio)
        
        self.merge_and_push = QCheckBox("Merge and Push")
        self.merge_and_push.setChecked(True)
        layout.addRow(self.merge_and_push)


        self.group_by_length = QCheckBox("Group sequences by length")
        self.group_by_length.setChecked(True)
        layout.addRow(self.group_by_length)

        self.save_steps = QLineEdit("100")
        layout.addRow(QLabel("Save Steps:"), self.save_steps)

        self.eval_steps = QLineEdit("100")
        layout.addRow(QLabel("Max Steps:"), self.max_steps)

        self.logging_steps = QLineEdit("10")
        layout.addRow(QLabel("Logging Steps:"), self.logging_steps)

        hbox_output_dir = QHBoxLayout()
        self.output_dir = QLineEdit("./results/LlamaChatOnlyBspSimp7b64")
        btn_output_dir = QPushButton("Browse")
        btn_output_dir.clicked.connect(lambda: self.output_dir.setText(self.browseForDirectory()))
        hbox_output_dir.addWidget(self.output_dir)
        hbox_output_dir.addWidget(btn_output_dir)
        layout.addRow(QLabel("Output Directory:"), hbox_output_dir)

        hbox_checkpoint = QHBoxLayout()
        self.checkpoint = QLineEdit("")
        btn_checkpoint = QPushButton("Browse")
        btn_checkpoint.clicked.connect(lambda: self.checkpoint.setText(self.browseForDirectory()))
        hbox_checkpoint.addWidget(self.checkpoint)
        hbox_checkpoint.addWidget(btn_checkpoint)
        layout.addRow(QLabel("Checkpoint Directory:"), hbox_checkpoint)

        hbox_train_eval_dir = QHBoxLayout()
        self.train_eval_dir = QLineEdit("./data/train_test_datasets/run_6_onlybsp_simple")
        btn_train_eval_dir = QPushButton("Browse")
        btn_train_eval_dir.clicked.connect(lambda: self.train_eval_dir.setText(self.browseForDirectory()))
        hbox_train_eval_dir.addWidget(self.train_eval_dir)
        hbox_train_eval_dir.addWidget(btn_train_eval_dir)
        layout.addRow(QLabel("Train and Eval Dataset Directory:"), hbox_train_eval_dir)


        widget.setLayout(layout)
        return widget


    def modelSettingsUI(self):
        widget = QWidget()
        layout = QFormLayout()

        # Create and add widgets for model settings
        self.model_name = QLineEdit("meta-llama/Llama-2-7b-chat-hf")
        layout.addRow(QLabel("Model Name:"), self.model_name)

        self.use_4bit = QCheckBox("Use 4bit precision")
        self.use_4bit.setChecked(True)
        layout.addRow(self.use_4bit)

        self.use_8bit = QCheckBox("Use 8bit precision")
        layout.addRow(self.use_8bit)

        self.use_lora = QCheckBox("Use LoRA")
        self.use_lora.setChecked(True)
        layout.addRow(self.use_lora)

        self.use_nested_quant = QCheckBox("Use Nested Quantization for 4bit models")
        layout.addRow(self.use_nested_quant)

        self.bnb_4bit_compute_dtype = QComboBox()
        self.bnb_4bit_compute_dtype.addItems(["float16", "float32"])  # Add additional data types if needed
        self.bnb_4bit_compute_dtype.setCurrentText("float16")
        layout.addRow(QLabel("Compute dtype for 4bit:"), self.bnb_4bit_compute_dtype)

        self.bnb_4bit_quant_type = QComboBox()
        self.bnb_4bit_quant_type.addItems(["nf4", "fp4"])  # Add other quant types if needed
        self.bnb_4bit_quant_type.setCurrentText("nf4")
        layout.addRow(QLabel("Quantization type (4bit):"), self.bnb_4bit_quant_type)

        self.fp16 = QCheckBox("Enable fp16 training")
        layout.addRow(self.fp16)

        self.bf16 = QCheckBox("Enable bf16 training")
        self.bf16.setChecked(True)
        layout.addRow(self.bf16)

        self.packing = QCheckBox("Use packing dataset")
        layout.addRow(self.packing)

        self.gradient_checkpointing = QCheckBox("Enable Gradient Checkpointing")
        self.gradient_checkpointing.setChecked(True)
        layout.addRow(self.gradient_checkpointing)

        
        self.wnb_project = QLineEdit("LLama")
        layout.addRow(QLabel("Weights and Biases Project Name:"), self.wnb_project)

        self.eval_collator = QComboBox()
        self.eval_collator.addItems(["all", "completion"])
        self.eval_collator.setCurrentText("completion")
        layout.addRow(QLabel("Evaluation Data Collator:"), self.eval_collator)

        self.train_collator = QComboBox()
        self.train_collator.addItems(["all", "completion"])
        self.train_collator.setCurrentText("all")
        layout.addRow(QLabel("Training Data Collator:"), self.train_collator)

        widget.setLayout(layout)
        return widget

    def startTraining(self):
        args = self.getScriptArgs()
        
        load_and_train(args)

    def getScriptArgs(self) -> ScriptArgs:
        return ScriptArgs(
            # Fetching TrainingConfig data
            local_rank=int(self.local_rank.text()),
            per_device_train_batch_size=int(self.per_device_train_batch_size.text()),
            per_device_eval_batch_size=int(self.per_device_eval_batch_size.text()),
            gradient_accumulation_steps=int(self.gradient_accumulation_steps.text()),
            learning_rate=float(self.learning_rate.text()),
            max_grad_norm=float(self.max_grad_norm.text()),
            weight_decay=float(self.weight_decay.text()),
            lora_alpha=int(self.lora_alpha.text()),
            lora_dropout=float(self.lora_dropout.text()),
            lora_r=int(self.lora_r.text()),
            max_seq_length=int(self.max_seq_length.text()),
            num_train_epochs=int(self.num_train_epochs.text()),
            optim=self.optim.currentText(),
            lr_scheduler_type=self.lr_scheduler_type.currentText(),
            max_steps=int(self.max_steps.text()),
            warmup_ratio=float(self.warmup_ratio.text()),
            group_by_length=self.group_by_length.isChecked(),
            save_steps=int(self.save_steps.text()),
            eval_steps=int(self.eval_steps.text()),
            logging_steps=int(self.logging_steps.text()),
            output_dir=self.output_dir.text(),
            checkpoint=self.checkpoint.text(),
            train_eval_dir=self.train_eval_dir.text(),
            merge_and_push=self.merge_and_push.isChecked(),

            # Fetching ModelConfig data
            model_name=self.model_name.text(),
            use_4bit=self.use_4bit.isChecked(),
            use_8bit=self.use_8bit.isChecked(),
            use_lora=self.use_lora.isChecked(),
            use_nested_quant=self.use_nested_quant.isChecked(),
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype.currentText(),
            bnb_4bit_quant_type=self.bnb_4bit_quant_type.currentText(),
            fp16=self.fp16.isChecked(),
            bf16=self.bf16.isChecked(),
            packing=self.packing.isChecked(),
            gradient_checkpointing=self.gradient_checkpointing.isChecked(),
            wnb_project=self.wnb_project.text(),
            eval_collator=self.eval_collator.currentText(),
            train_collator=self.train_collator.currentText()
        )



window = AppWindow()
sys.exit(app.exec_())