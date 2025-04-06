import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
import numpy as np

# 假設：每個藥品、病情都用數字ID表示
num_drugs = 5000      # 假設5000種藥品
num_conditions = 2000 # 假設2000種病情
embedding_dim = 64    # Embedding向量維度

# 藥品輸入
drug_input = Input(shape=(1,), name='drug_input')
drug_embedding = Embedding(input_dim=num_drugs, output_dim=embedding_dim, name='drug_embedding')(drug_input)
drug_embedding = Flatten()(drug_embedding)

# 病情輸入
condition_input = Input(shape=(1,), name='condition_input')
condition_embedding = Embedding(input_dim=num_conditions, output_dim=embedding_dim, name='condition_embedding')(condition_input)
condition_embedding = Flatten()(condition_embedding)

# 年齡輸入
age_input = Input(shape=(1,), name='age_input')

# 性別輸入（0=男，1=女）
gender_input = Input(shape=(1,), name='gender_input')

# 合併特徵
merged = Concatenate()([drug_embedding, condition_embedding, age_input, gender_input])

# 全連接層
x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)

# 多任務輸出
interaction_output = Dense(1, activation='sigmoid', name='interaction_output')(x)  # 交互作用
adverse_effect_output = Dense(1, activation='sigmoid', name='adverse_effect_output')(x)  # 副作用風險
contraindication_output = Dense(1, activation='sigmoid', name='contraindication_output')(x)  # 禁忌警告

# 建立模型
model = Model(inputs=[drug_input, condition_input, age_input, gender_input],
              outputs=[interaction_output, adverse_effect_output, contraindication_output])

# 編譯模型
model.compile(
    optimizer='adam',
    loss={
        'interaction_output': 'binary_crossentropy',
        'adverse_effect_output': 'binary_crossentropy',
        'contraindication_output': 'binary_crossentropy'
    },
    metrics=['accuracy']
)

model.summary()

# ======== 假資料（測試用）========
# 藥品ID, 病情ID, 年齡, 性別
X_drug = np.random.randint(0, num_drugs, size=(1000, 1))
X_condition = np.random.randint(0, num_conditions, size=(1000, 1))
X_age = np.random.randint(18, 90, size=(1000, 1))
X_gender = np.random.randint(0, 2, size=(1000, 1))

# 標籤：交互作用、有副作用、有禁忌
y_interaction = np.random.randint(0, 2, size=(1000, 1))
y_adverse_effect = np.random.randint(0, 2, size=(1000, 1))
y_contraindication = np.random.randint(0, 2, size=(1000, 1))

# 訓練模型
model.fit(
    [X_drug, X_condition, X_age, X_gender],
    [y_interaction, y_adverse_effect, y_contraindication],
    epochs=10,
    batch_size=32
)

