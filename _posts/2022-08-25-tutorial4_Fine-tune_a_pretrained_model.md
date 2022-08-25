---
title: "[Huggingface 🤗 Transformers Tutorial] 4. Fine-tune a pretrained model"
excerpt: "🤗 Transformers를 이용하여 pretrained model을 fine-tuning하는 방법을 배워보고 sentiment analysis(감정 분석) task를 간단하게 수행해봅니다."
toc: true
toc_sticky: true
categories:
    - nlp
tags:
    - huggingface
    - transformers
sidebar:
    nav: sidebarTotal
---

※ 이 글의 원문은 [이 곳](https://huggingface.co/docs/transformers/training#finetune-a-pretrained-model)에서 확인할 수 있습니다.
※ 모든 글의 내용을 포함하지 않으며 새롭게 구성한 내용도 포함되어 있습니다.

🤗 Transformers에 올라와있는 pretrained 모델들을 specific task에 맞게 train하는 방법을 배워보도록 하겠습니다.
* fine-tuning


```python
!pip install transformers
!pip install datasets
```

활용할 데이터셋은 [nsmc(naver sentiment movie corpus)](https://huggingface.co/datasets/nsmc)입니다.
* 영화 리뷰 댓글을 이용해 감정 분류하는 목적으로 제작된 데이터셋입니다.
* 0(negative), 1(postiive)


```python
from datasets import load_dataset

nsmc = load_dataset('nsmc')
```


```python
nsmc
```




    DatasetDict({
        train: Dataset({
            features: ['id', 'document', 'label'],
            num_rows: 150000
        })
        test: Dataset({
            features: ['id', 'document', 'label'],
            num_rows: 50000
        })
    })



train set은 150000개, test set은 50000개로 이루어져있는 것을 확인할 수 있습니다.

전체 데이터를 학습하려면 오래 걸리기 때문에 학습 데이터 2000개, 테스트 데이터 2000개를 random sampling하겠습니다.


```python
train_data = nsmc['train'].shuffle(seed=42).select(range(2000))
test_data = nsmc['test'].shuffle(seed=42).select(range(2000))
```


```python
train_data
```




    Dataset({
        features: ['id', 'document', 'label'],
        num_rows: 2000
    })




```python
test_data
```




    Dataset({
        features: ['id', 'document', 'label'],
        num_rows: 2000
    })



영화 댓글 감정 분류 작업을 실습해볼 것입니다. 분류를 위한 model과 tokenizer를 load합니다.
* 문장이 어떤 감정에 해당하는 지 분류하는 것이기 때문에 Sequence Classification을 위한 AutoClass인 AutoModelForSequenceClassification을 사용합니다.
* model과 tokenizer로 ['bert-base-multilingual-cased'](https://huggingface.co/bert-base-multilingual-cased)을 사용합니다.


```python
MODEL_NAME = 'bert-base-multilingual-cased'
```


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

tokenizer가 정상적으로 동작하는지 테스트합니다.


```python
tokenizer.tokenize(train_data['document'][0])
```




    ['For',
     'Carl',
     '.',
     '칼',
     '세',
     '##이',
     '##건',
     '##으로',
     '시',
     '##작',
     '##해서',
     '칼',
     '세',
     '##이',
     '##건',
     '##으로',
     '끝',
     '##난',
     '##다',
     '.']



먼저 data의 document를 모두 encoding합니다.


```python
train_encoding = tokenizer(
    train_data['document'],
    return_tensors='pt',
    padding=True,
    truncation=True
)

test_encoding = tokenizer(
    test_data['document'],
    return_tensors='pt',
    padding=True,
    truncation=True
)
```


```python
len(train_encoding['input_ids']), len(test_encoding['input_ids'])
```




    (2000, 2000)



학습을 위한 데이터셋을 구성해보겠습니다.

BERT를 이용해 영화 댓글의 감정을 분류하는 작업을 할 것입니다. 다음과 같은 요소가 필요합니다.
1. input_ids
2. token_type_ids
3. attention_mask
4. labels

1~3번은 tokenizer를 이용해 이미 만들었습니다. labels는 random sampled된 dataset의 label을 이용하면 됩니다.

PyTorch의 Dataset 클래스를 이용해 학습과 검증을 위한 데이터셋을 만듭니다.


```python
import torch
from torch.utils.data import Dataset


class NSMCDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        data = {key: val[idx] for key, val in self.encoding.items()}
        data['labels'] = torch.tensor(self.labels[idx]).long()

        return data

    def __len__(self):
        return len(self.labels)
```


```python
train_set = NSMCDataset(train_encoding, train_data['label'])
test_set = NSMCDataset(test_encoding, test_data['label'])
```


```python
train_set[0]
```




    {'input_ids': tensor([  101, 11399, 12225,   119,  9788,  9435, 10739, 71439, 11467,  9485,
             38709, 70146,  9788,  9435, 10739, 71439, 11467,  8977, 33305, 11903,
               119,   102,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0]),
     'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     'labels': tensor(1)}




```python
test_set[0]
```




    {'input_ids': tensor([  101, 14796, 27728, 10230,   106,   102,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0]),
     'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0]),
     'attention_mask': tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0]),
     'labels': tensor(0)}



## Trainer 클래스로 학습하기

🤗 Transformers는 model 학습을 위해 TrainingArguments, Trainer 클래스를 제공합니다.
* TrainingArguments Trainer를 위한 Argument 클래스라고 보면됩니다.

TrainingArguments, Trainer를 이용하면 training option, logging, gradient accumulation, mixed precision을 간단하게 설정해 학습, 평가를 모두 진행할 수 있습니다.



```python
from transformers import TrainingArguments, Trainer
```

먼저 Training에 필요한 argument를 정의하겠습니다.
* 아래 사용한 parameter보다 다양한 parameter가 존재하니 [TrainingArguments](https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/trainer#transformers.TrainingArguments)를 참고하면 좋을 것 같습니다.


```python
training_args = TrainingArguments(
    output_dir = './outputs', # model이 저장되는 directory
    logging_dir = './logs', # log가 저장되는 directory
    num_train_epochs = 10, # training epoch 수
    per_device_train_batch_size=32,  # train batch size
    per_device_eval_batch_size=32,   # eval batch size
    logging_steps = 50, # logging step, batch단위로 학습하기 때문에 epoch수를 곱한 전체 데이터 크기를 batch크기로 나누면 총 step 갯수를 알 수 있다.
    save_steps= 50, # 50 step마다 모델을 저장한다.
    save_total_limit=2 # 2개 모델만 저장한다.
)
```

GPU 학습을 위해 device를 cude로 설정합니다.

TrainingArguments를 이용해 Trainer를 만듭니다.


```python
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

device
```




    device(type='cuda')



Trainer 클래스는 별도의 metric을 제공해주지않기 때문에 별도의 함수를 통해 계산을 따로 해주어야합니다.

accuracy와 f1 score를 계산하기위한 compute_metrics 함수를 만듭니다.
* 해당 함수는 인자를 통해 EvalPrediction 객체를 넘겨 받습니다.
* EvalPrediction은 predictions와 label_ids를 가집니다.
    * predictions: model의 예측값
    * label_ids: label 값
* datasets에서 제공하는 load_metric()을 이용해 accuracy와 f1 score를 계산합니다.



```python
from datasets import load_metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    m1 = load_metric('accuracy')
    m2 = load_metric('f1')

    acc = m1.compute(predictions=preds, references=labels)['accuracy']
    f1 = m2.compute(predictions=preds, references=labels)['f1']

    return {'accuracy':acc, 'f1':f1}
```


```python
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set, # 학습 세트
    eval_dataset=test_set, # 테스트 세트
    compute_metrics=compute_metrics # metric 계산 함수
)
```

model을 학습합니다.


```python
trainer.train()
```

    ***** Running training *****
      Num examples = 2000
      Num Epochs = 10
      Instantaneous batch size per device = 32
      Total train batch size (w. parallel, distributed & accumulation) = 32
      Gradient Accumulation steps = 1
      Total optimization steps = 630





<table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>0.697200</td>
    </tr>
    <tr>
      <td>100</td>
      <td>0.631900</td>
    </tr>
    <tr>
      <td>150</td>
      <td>0.509600</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.401600</td>
    </tr>
    <tr>
      <td>250</td>
      <td>0.314900</td>
    </tr>
    <tr>
      <td>300</td>
      <td>0.210400</td>
    </tr>
    <tr>
      <td>350</td>
      <td>0.155300</td>
    </tr>
    <tr>
      <td>400</td>
      <td>0.105700</td>
    </tr>
    <tr>
      <td>450</td>
      <td>0.101600</td>
    </tr>
    <tr>
      <td>500</td>
      <td>0.058900</td>
    </tr>
    <tr>
      <td>550</td>
      <td>0.046800</td>
    </tr>
    <tr>
      <td>600</td>
      <td>0.041600</td>
    </tr>
  </tbody>
</table>

    TrainOutput(global_step=630, training_loss=0.26070984389100754, metrics={'train_runtime': 507.216, 'train_samples_per_second': 39.431, 'train_steps_per_second': 1.242, 'total_flos': 1223055296400000.0, 'train_loss': 0.26070984389100754, 'epoch': 10.0})



model을 평가합니다.


```python
trainer.evaluate()
```

    ***** Running Evaluation *****
      Num examples = 2000
      Batch size = 32
    {'eval_loss': 1.3575892448425293,
     'eval_accuracy': 0.762,
     'eval_f1': 0.7586206896551724,
     'eval_runtime': 16.4764,
     'eval_samples_per_second': 121.386,
     'eval_steps_per_second': 3.824}


데이터 수가 적기 때문에 accuracy가 약 76정도 나온 것을 확인할 수 있습니다.

## PyTorch Native 방식으로 학습하기

TrainingArguments와 Trainer를 사용하지않고 학습하는 방법입니다.

먼저 DataLoader를 만들겠습니다. batch size는 이전과 동일하게 32입니다.


```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)
```

다음으로는 train 함수를 구성할 것입니다.

train 함수를 구성하려면 model의 output이 어떻게 구성되었는지 확인해야합니다.

dummy 데이터를 이용해 model의 output이 어떻게 나오는 지 확인해봅니다.


```python
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
```


```python
dummy = tokenizer(train_data['document'][0], return_tensors='pt')

model(**dummy)
```




    SequenceClassifierOutput(loss=None, logits=tensor([[-0.0045,  0.1503]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)



model의 output에서 주목할 것은 loss와 logits입니다. logits은 model의 예측값을 가리킵니다. loss는 말그대로 loss 값을 가리킵니다.
* label이 없기 때문에 loss가 계산되지 않았습니다.

이 2가지 정보를 이용해 train 함수를 구성하겠습니다.

train 함수에는 이전과 동일하게 accuracy와 f1 score를 계산하는 것도 추가하겠습니다.


```python
from tqdm.notebook import tqdm
from datasets import load_metric

def train(epoch, model, dataloader, optimizer, device):
    model.to(device)

    m1 = load_metric('accuracy')
    m2 = load_metric('f1')

    for e in range(1, epoch+1):
        total_loss = 0.
        preds = []
        labels = []
        progress_bar = tqdm(dataloader, desc=f'TRAIN - EPOCH {e} |')
        for data in progress_bar:
            data = {k:v.to(device) for k, v in data.items()}
            output = model(**data)
            current_loss = output.loss

            total_loss += current_loss
            preds += list(output.logits.argmax(-1))
            labels += list(data['labels'].detach().cpu().numpy())

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            progress_bar.set_description(f'TRAIN - EPOCH {e} | current-loss: {current_loss:.4f}')
        
        acc = m1.compute(predictions=preds, references=labels)['accuracy']
        f1 = m2.compute(predictions=preds, references=labels)['f1']
        avg = total_loss / len(dataloader)

        print('='*64)
        print(f"TRAIN - EPOCH {e} | LOSS: {avg:.4f} ACC: {acc:.4f} F1: {f1:.4f}")
        print('='*64)

```

evaluate 함수도 train함수와 비슷하게 구성하지만 epoch이 없고 backward 과정이 없습니다.


```python
def evaluate(model, dataloader, device):
    model.to(device)

    m1 = load_metric('accuracy')
    m2 = load_metric('f1')

    total_loss = 0.
    preds = []
    labels = []
    progress_bar = tqdm(dataloader, desc=f'EVAL |')
    for data in progress_bar:
        data = {k:v.to(device) for k, v in data.items()}

        with torch.no_grad():
            output = model(**data)
    
        current_loss = output.loss

        total_loss += current_loss
        preds += list(output.logits.argmax(-1))
        labels += list(data['labels'].detach().cpu().numpy())

        progress_bar.set_description(f'EVAL | current-loss: {current_loss:.4f}')
    
    acc = m1.compute(predictions=preds, references=labels)['accuracy']
    f1 = m2.compute(predictions=preds, references=labels)['f1']
    avg = total_loss / len(dataloader)

    print('='*64)
    print(f"EVAL | LOSS: {avg:.4f} ACC: {acc:.4f} F1: {f1:.4f}")
    print('='*64)
```

함수를 구성했으므로 본격적으로 PyTorch를 이용해 학습을 진행합니다.




```python
from torch.optim import AdamW

# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
```


```python
train(10, model, train_loader, optimizer, device)
```


    ================================================================
    TRAIN - EPOCH 1 | LOSS: 0.6821 ACC: 0.5530 F1: 0.4659
    ================================================================


    ================================================================
    TRAIN - EPOCH 2 | LOSS: 0.5740 ACC: 0.6825 F1: 0.6315
    ================================================================


    ================================================================
    TRAIN - EPOCH 3 | LOSS: 0.4387 ACC: 0.8055 F1: 0.7958
    ================================================================


    ================================================================
    TRAIN - EPOCH 4 | LOSS: 0.3261 ACC: 0.8760 F1: 0.8704
    ================================================================


    ================================================================
    TRAIN - EPOCH 5 | LOSS: 0.2744 ACC: 0.8895 F1: 0.8873
    ================================================================


    ================================================================
    TRAIN - EPOCH 6 | LOSS: 0.1774 ACC: 0.9405 F1: 0.9399
    ================================================================


    ================================================================
    TRAIN - EPOCH 7 | LOSS: 0.1397 ACC: 0.9545 F1: 0.9544
    ================================================================


    ================================================================
    TRAIN - EPOCH 8 | LOSS: 0.0979 ACC: 0.9695 F1: 0.9691
    ================================================================


    ================================================================
    TRAIN - EPOCH 9 | LOSS: 0.0688 ACC: 0.9750 F1: 0.9747
    ================================================================


    ================================================================
    TRAIN - EPOCH 10 | LOSS: 0.0436 ACC: 0.9845 F1: 0.9844
    ================================================================


결과를 빨리 내기 위해 random sampling된 2000개의 데이터로 학습한 것이기 때문에 overfitting이 발생한 것을 짐작할 수 있습니다.

원본 데이터를 그대로 이용하면 좋은 결과를 얻을 수 있을 것입니다.

model 평가를 진행하겠습니다.
* Trainer를 활용해 학습을 진행한 것과 비슷한 결과를 얻은 것을 확인할 수 있습니다.


```python
evaluate(model, test_loader, device)
```


    ================================================================
    EVAL | LOSS: 1.0961 ACC: 0.7610 F1: 0.7675
    ================================================================


🤗 huggingface의 pretrained model을 fine-tuning하는 방법을 간단하게 알아보았습니다.

이번에는 pretrained model을 사용하는 방법을 빠르게 알아보기 위해 Sequence Classfication 중 하나인 Sentiment analysis(감정 분석) task를 진행해보았습니다.

다음 번부터는 🤗 huggingface의 How-To-Guides를 살펴보면서 NLP 분야의 downstream task를 실습해보는시간을 진행하려고 합니다.

기회가 되면 BERT, GPT 외에 다른 모델도 한 번 리뷰해보겠습니다.
