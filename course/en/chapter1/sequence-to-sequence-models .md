# sequence-to-sequence-models

### **编码器-解码器模型（Encoder-Decoder Models）详解**

也称为 **序列到序列模型（Sequence-to-Sequence, Seq2Seq）**，这类模型同时使用 Transformer 的 **编码器（Encoder）** 和 **解码器（Decoder）** 部分，适用于需要**根据输入生成新序列**的任务。

---

### **1. 核心架构与工作原理**

### **（1）编码器（Encoder）**

- **功能**：将输入序列（如句子）编码为**上下文相关的特征表示**。
- **注意力机制**：可访问输入的所有词（双向注意力），全面理解语义。
    - *例如*：翻译时，编码器会分析整个源句子（如英文句子）。

### **（2）解码器（Decoder）**

- **功能**：基于编码器的输出，**自回归生成**目标序列（如翻译结果）。
- **注意力机制**：
    - **编码器-解码器注意力**：可访问编码器的全部输出。
    - **自注意力**：仅能访问已生成的词（单向注意力，防止“偷看”未来词）。
    - *例如*：生成法语翻译时，解码器逐步输出词，且每一步只能看到之前的词。

---

### **2. 预训练方法**

不同于纯编码器或解码器模型，Seq2Seq 的预训练目标更复杂，例如：

- **T5 的 Span Masking**：
    - 随机遮盖文本片段（可能包含多个词），替换为特殊标记 `<mask>`。
    - 目标：预测被遮盖的完整文本（类似“填空”）。
- **BART 的噪声还原**：
    - 对输入添加多种噪声（如删除、打乱词序），让模型还原原始文本。

---

### **3. 典型任务与应用**

Seq2Seq 模型擅长**输入到输出的序列转换**，例如：

| **任务** | **示例** |
| --- | --- |
| **机器翻译** | 输入：“Hello” → 输出：“Bonjour” |
| **文本摘要** | 输入：“长文章” → 输出：“简短摘要” |
| **生成式问答** | 输入：“什么是光合作用？” → 输出：“光合作用是植物利用阳光制造能量的过程。” |
| **对话生成** | 输入：“你好！” → 输出：“你好，有什么可以帮您？” |

---

### **4. 代表模型**

| **模型** | **特点** |
| --- | --- |
| **T5** | 将所有任务统一为“文本到文本”格式（如翻译任务前缀加`translate English to French:`）。 |
| **BART** | 结合双向编码器+自回归解码器，擅长文本生成与重构。 |
| **mBART** | 多语言版BART，支持50+语言的翻译和生成。 |
| **Marian** | 轻量级神经机器翻译模型，专为低资源语言优化。 |

---

### **5. 与纯编码器/解码器模型的对比**

| **模型类型** | **架构** | **典型任务** | **代表模型** |
| --- | --- | --- | --- |
| **Encoder-only** | 纯编码器 | 文本分类、实体识别 | BERT, RoBERTa |
| **Decoder-only** | 纯解码器 | 文本生成、对话 | GPT 系列 |
| **Encoder-Decoder** | 编码器+解码器 | 翻译、摘要、生成式问答 | T5, BART |

---

### **6. 使用示例（T5 翻译）**

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

input_text = "translate English to French: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# 输出: "Bonjour, comment allez-vous?"

```

---

### **7. 为什么选择 Seq2Seq 模型？**

- **输入输出灵活**：可处理变长输入和输出序列。
- **任务通用性**：通过任务前缀（如`summarize:`）适配多种任务。
- **生成质量高**：解码器的自回归特性保证生成文本的连贯性。

**适用场景**：需同时理解输入并生成复杂输出的任务（如翻译需先理解源语言，再生成目标语言）。