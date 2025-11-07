
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def run_training():
    # 1. 데이터 로드
    try:
        train_df = pd.read_csv('data/raw/train.csv')
        # '일반 대화' 클래스를 4로 라벨링
        # 이 부분은 프로젝트의 필요에 따라 수정해야 합니다.
        # 예시: 일반 대화 데이터를 불러와서 합치는 과정
        # normal_df = pd.read_csv('data/processed/daily_conversations_200.csv')
        # normal_df['class'] = 4 # 일반 대화 클래스
        # train_df = pd.concat([train_df, normal_df])
    except FileNotFoundError:
        print("데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: 'data/raw/train.csv'")
        return

    # 클래스 라벨을 숫자로 변환 (필요시)
    labels = {'협박 대화': 0, '갈취 대화': 1, '직장 내 괴롭힘 대화': 2, '기타 괴롭힘 대화': 3}
    train_df['label'] = train_df['class'].map(labels)
    train_df = train_df.dropna(subset=['label'])
    train_df['label'] = train_df['label'].astype(int)

    # 2. 데이터셋 분할 및 Hugging Face Dataset 형식으로 변환
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['conversation'].tolist(),
        train_df['label'].tolist(),
        test_size=0.2,
        random_state=42
    )

    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

    raw_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # 3. 토크나이저 및 모델 설정
    model_name = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4) # 클래스 갯수에 맞게 수정

    # 4. 토큰화 함수 정의
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # 5. 훈련 인자(Arguments) 설정
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 6. Trainer 설정 및 훈련 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    print("모델 훈련을 시작합니다.")
    trainer.train()
    print("모델 훈련이 완료되었습니다.")

    # 7. 훈련된 모델 저장
    trainer.save_model("./results/best_model")
    tokenizer.save_pretrained("./results/best_model")
    print("훈련된 모델이 './results/best_model'에 저장되었습니다.")

if __name__ == "__main__":
    run_training()
