from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """

  model_name_or_path: str = field(
      metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  config_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
      default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
  )


@dataclass
class MultimodalDataTrainingArguments:
  """
  Arguments pertaining to how we combine tabular features
  Using `HfArgumentParser` we can turn this class
  into argparse arguments to be able to specify them on
  the command line.
  """

  data_path: str = field(metadata={
                            'help': 'the path to the csv file containing the dataset'
                        })
  column_info_path: str = field(
      default=None,
      metadata={
          'help': 'the path to the json file detailing which columns are text, categorical, numerical, and the label'
  })

  column_info: dict = field(
      default=None,
      metadata={
          'help': 'a dict referencing the text, categorical, numerical, and label columns'
                  'its keys are text_cols, num_cols, cat_cols, and label_col'
  })

  categorical_encode_type: str = field(default='ohe',
                                        metadata={
                                            'help': 'sklearn encoder to use for categorical data',
                                            'choices': ['ohe', 'binary', 'label', 'none']
                                        })
  numerical_transformer_method: str = field(default='yeo_johnson',
                                            metadata={
                                                'help': 'sklearn numerical transformer to preprocess numerical data',
                                                'choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']
                                            })
  task: str = field(default="classification",
                    metadata={
                        "help": "The downstream training task",
                        "choices": ["classification", "regression"]
                    })

  mlp_division: int = field(default=4,
                            metadata={
                                'help': 'the ratio of the number of '
                                        'hidden dims in a current layer to the next MLP layer'
                            })
  combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat',
                                    metadata={
                                        'help': 'method to combine categorical and numerical features, '
                                                'see README for all the method'
                                    })
  mlp_dropout: float = field(default=0.1,
                              metadata={
                                'help': 'dropout ratio used for MLP layers'
                              })
  numerical_bn: bool = field(default=True,
                              metadata={
                                  'help': 'whether to use batchnorm on numerical features'
                              })
  use_simple_classifier: str = field(default=True,
                                      metadata={
                                          'help': 'whether to use single layer or MLP as final classifier'
                                      })
  mlp_act: str = field(default='relu',
                        metadata={
                            'help': 'the activation function to use for finetuning layers',
                            'choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']
                        })
  gating_beta: float = field(default=0.2,
                              metadata={
                                  'help': "the beta hyperparameters used for gating tabular data "
                                          "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
                              })

  def __post_init__(self):
      assert self.column_info != self.column_info_path
      if self.column_info is None and self.column_info_path:
          with open(self.column_info_path, 'r') as f:
              self.column_info = json.load(f)