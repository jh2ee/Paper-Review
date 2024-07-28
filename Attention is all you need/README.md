# Abstract

기존의 Sequence transduction model 들은 encoder 와 decoder를 포함하는 복잡한 recurrent 또는 CNN 구조에 기반한다. 본 논문에서는 Attention mechanism 만을 사용한 Transformer 구조를 제안한다.

번역 task 들을 통해 우수한 병렬 처리 성능과 학습 시간의 단축을 보인다. BLEU 평가 방법을 통해 기존의 task 에 대해 우수한 성적을 입증했다.

# Instroduction

RNN, LSTM, GRU 는 sequence modeling, transduction problem 들에 대해 자리잡은 모델들이지만 이전 결과를 입력으로 받는 순차적 특성으로 인해 병렬 처리에 어려움이 있다. 최근의 연구에서 conditional computation에 대한 성능 개선이 이루어졌지만 여전히 본질적인 sequential computation 문제는 존재한다.

기존의 Attention mechanism 은 RNN 과 함께 사용되었다. 그러나 Transformer 는 recurrence를 제거해 병렬성을 향상시켰다.

# Background

Self-attention 은 단일 sequence에 대해 **위치에 따라 다른 attention 을 수행**하는 것으로 스스로에 대해attention 을 수행하는 것이다.

Transformer 는 RNN 또는 convolution 을 사용하지 않고 self-attention 으로 input 과 output 을 계산하는 첫 transduction model 이다.

# Model Architecture

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/542b861c-36a8-4051-84e5-8804b6728dba/d85562bd-05fd-4d06-9cff-b7bdfaee3a06/Untitled.png)

Transformer 는 attention 만으로 Encoder 와 Decoder 를 가진 구조이다.

## Encoder

Encoder는 6개의 독립적인 layer (Encoder Block) 들로 구성되어 있으며 각 layer는 2개의 sub-layer 를 갖는다.

첫번째로는 multi-head self-attention, 두 번째는 position-wise fully feed-forward network 이다.

두 sub-layer 간에는 residual connection 과 normalization 을 수행한다. 이는 Gradient vanishing/exploding 문제를 완화하고 layer 간의 정보 전달을 돕는다.

논문에서 모델의 모든 sub-layer 의 차원 $d_{model}$ 은 512이다.

## Decoder

Decoder 또한 6개의 독립적인 layer (Decoder Block) 들로 구성되어 있으며 Encoder 에 존재하는 두 sub-layer 뿐 아니라 Encoder 의 output 과 multi-head attention 을 수행하는 sub-layer 가 존재한다.

마찬가지로 세 sub-layer 간 residual connection 과 normalization 을 수행한다.

## Attention

Attention 은 query 와 key-value 쌍을 output 에 mapping 하는 것으로 묘사될 수 있다. 이때 query, key, value, output 은 모두 vector 의 형태이다.

Output 은 각각의 weighted sum(가중치) 로 계산되며 각 값에 할당되는 weight 는 query 와 key의 compatibility function 에 의해 계산된다.

<aside>
💡 쉽게 말해 attention 의 목표는 value 를 통해 **weight sum 을 계산**하는 것이며 value 의 weight 는 **query 와 key 의 유사도를 통해 결정**된다.

</aside>

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/542b861c-36a8-4051-84e5-8804b6728dba/45ddcf3b-525a-4ed7-badd-ff9f814cd5e2/Untitled.png)

Transformer 구조에서 사용하는 Attention 은 위와 같은 구조를 가진다. 

수식에 있어 query 와 key 의 차원을 $d_k$, value 의 차원을 $d_v$ 라고 하자. 논문에서는 $d_k,d_v$ 는 64로 설정했다.

Scaled Dot-Product Attention 은 좌측 그림과 같이 계산된다. Attention mechanism 에서는 query 의 set 을 동시에 matrix 로 계산한다. key 와 value 또한 마찬가지로 계산되는데 수식으로 나타내면 다음과 같다.

$$
Attention(Q,K,V) = softmax(\cfrac{QK^T}{\sqrt{d_k}})V
$$

query 의 set 인 Q 와 key 의 set 인 K 를 행렬곱 연산하고 이를 차원의 루트를 씌운 값으로 나누어 준다. 나누어 주는 이유는 차원이 커졌을 경우 행렬곱 값들이 커지는 것을 방지하기 위함이다. 그림에서의 **Scale** 과정으로 큰 값에 대해서는 softmax 함수의 output 을 안정화하고 작은 값에 대해서는 vanishing 문제를 방지하는 역할을 한다.

이런 Attention 구조가 여럿 모여 만들어진 것이 우측의 **Multi-Head Attention** 이다. 

Multi-Head Attention 은 Q, K, V 에 대해 Linear 과정을 거친 후 Scaled Dot-Product Attention 을 수행하고 이 값들을 다시 concatenate, Linear 과정을 수행해 output 을 생성한다.

<aside>
💡 **그렇다면 concatenate 과정은 왜 필요할까?**

그 이유는 차원을 head 수로 쪼개어 h 번 학습을 진행하기 때문이다. Multi-Head Attention 라는 이름이 붙은 것도 attention 을 여러 head 로 나누어 수행하기 때문이다.
attention 을 한번에 수행하지 않고 나누어 수행하는 이유는 multi-head 가 서로 다른 **Representation Subspaces** 를 학습하여 다양한 유형의 종속성을 가져 다양한 정보를 결합할 수 있기 때문이다. 문장 번역 task 를 예로 들면 한 문장에 대해 문장 타입 구별, 명사, 관계, 강조 등 다양한 유형에 대해 집중할 수 있다. 논문에서는 8개의 head 를 구성했다.

</aside>

Multi-Head Attention 을 수식으로 표현하면 다음과 같다.

$$
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O \\ where head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)
$$

## Attention Models

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/542b861c-36a8-4051-84e5-8804b6728dba/d85562bd-05fd-4d06-9cff-b7bdfaee3a06/Untitled.png)

Transformer 에서는 3가지의 attention model 이 사용되었다.

### Encoder-Decoder Attention

이 layer 에서는 query 는 이전 decoder layer(block) 에서 받고 key, value 는 encoder 의 output 에서 받는다. 위 그림에서 Decoder 의 2번째 layer, Multi-Head Attention layer 에 해당한다. 이 layer 를 통해 Decoder 의 sequence 들이 Encoder 의 어떤 sequence 와 연관을 갖는지 학습할 수 있다.

### Encoder Self-Attention

그림에서 Encoder 의 Multi-Head Attention layer 에 해당한다. Encoder 의 self-attention 은 query, key, value 모두 encoder 에서 가져오며 이전의 encoder layer(block) 의 output 을 input 으로 받아 이전 layer 의 모든 position 에 대해 학습 가능하다.

### Decoder Self-Attention

그림에서 Decoder 의 Masked Multi-Head Attention layer 에 해당한다. Encoder 의 self-attention layer 와 전반적으로 비슷한 기능을 하지만 decoder 에서는 sequence model 의 Auto-Regressive property 보존을 위해 이후에 나올 단어를 참조하지 않는다. 

예를 들어 I love dog 라는 문장이 있을 때의 query 가 love 라면 dog 에 대한 key 값은 참조하지 않는 것이다. layer 이름의 Masked 가 붙은 것도 이처럼 참조하지 않는 토큰(아직 주어지지 않은 쿼리)에 대해 마스킹을 진행하기 때문이다.

## Position-wise Feed-Forward Networks

Encoder 와 Decoder 에 모두 포함된 **fully-connected feed-forward layer** 이다. 2개의 Linear layer 사이에 RELU activation 을 사용한 구조로 논문에서는 input, output 의 차원은 동일하게 $d_{model}=512$ 로 두었고 Linear 사이의 차원인 $d_{ff}=2048$ 로 설정했다.

이 layer 를 통해 비선형 변화를 거치면서 position 정보를 강화하고 특징을 추출해 sequence 를 효과적으로 처리, 학습시킬 수 있도록 한다.

## Embeddings and Softmax

### Embeddings

다른 sequence transduction model 들과 유사하게 input token, output token 들을 $d_{model}$ 차원의 vector 로 변환하기 위해 학습된 embedding 을 사용한다. 

input embedding 과 output embedding 은 $d_{model}$ 차원의 embedding layer 를 거치고 두 embedding 은 linear transform 으로 같은 weight matrix 를 공유한다. 

weight 에는 $\sqrt{d_{model}}$ 을 곱한다.

### Softmax

Decoder 의 output 출력 과정에서 linear 를 거친 후 softmax 를 취해 다음 token 의 확률값을 계산한다.

## Positional Encoding

Transformer 모델이 recurrence 나 convolution 을 포함하지 않기에 sequence 의 순서에 대한 정보를 만들기 위해 sequence 에서 token 의 상대적 또는 절대적 위치에 대한 정보를 주입해야 한다. 이 과정을 수행하기 위해 positional encoding 이 사용된다.

논문에서는 sin 곡선의 상대적 위치에 대한 정보를 embedding 과 동일한 차원에 위치시키는 방법을 사용했다. Embedding 차원인 $d_{model}$ 과 동일한 차원의 sin 그래프를 단순히 더해주면 된다. 수식으로 표현하면 아래와 같다.

$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}}) \\ or \space PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})
$$

이때 pos 는 position 의 위치, i 는 hidden dimension 의 위치이다.

# Reference

[https://lcyking.tistory.com/entry/논문리뷰-Attention-is-All-you-need의-이해](https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Attention-is-All-you-need%EC%9D%98-%EC%9D%B4%ED%95%B4)

https://velog.io/@tobigs-nlp/Attention-is-All-You-Need-Transformer

---

# 구현

<aside>
💡 **본 코드는 [동빈나, [딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)](https://www.youtube.com/watch?v=AA621UofTUA&t=5s) 를 기반으로 작성되었음**

</aside>

> **실습 코드 :** https://colab.research.google.com/drive/1pgQW82jfs0Q3d_pMXq-Tg-TRdj2wprBP?usp=sharing
> 

## Multi-Head Attention Layer

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
    super().__init__()

    self.hidden_dim = hidden_dim # embedding 차원
    self.n_heads = n_heads # self attention의 수
    self.head_dim = hidden_dim // n_heads # 각 head에서의 embedding 차원

    # q, k, v layer
    self.fc_qkv = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어

    # attention output layer
    self.fc_o = nn.Linear(hidden_dim, hidden_dim)

    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self, q_in, k_in, v_in, mask=None):
    batch_size = q_in.shape[0]

    def transform_qkv(x):
	    # input의 차원 변경하는 함수
      # [batch_size, qkv_len, hidden_dim] -> [batch_size, n_heads, qkv_len, head_dim]
      out = self.fc_qkv(x)
      out = out.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
      return out

    # Q, K, V : [batch_size, n_heads, qkv_len, head_dim]
    Q = transform_qkv(q_in)
    K = transform_qkv(k_in)
    V = transform_qkv(v_in)

    # calculate attention
    # Attention = softmax( (Q*K_transposed) / sqrt(d_k) ) * V
    scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    # attention : [batch_size, n_heads, query_len, key_len]
    attention = torch.matmul(Q, K.permute(0,1,3,2)) 
    attention = attention / scale
    attention = torch.softmax(attention, dim=-1)

    out = torch.matmul(attention, V) # out : [batch_size, n_heads, query_len, head_dim]
    out = out.permute(0,2,1,3).contiguous() # out : [batch_size, query_len, n_heads, head_dim]
    out = out.view(batch_size, -1, self.hidden_dim) # out : [batch_size, query_len, hidden_dim]
    
    out = self.fc_qkv(out) # nn.Linear(self.hidden_dim, self.hidden_dim)

    return out, attention # attention : graph 그리기 위해 반환, 반환하지 않아도 됨

```

## Position-Wise Feed Foward Layer

```python
class PositionWiseFF(nn.Module):
  def __init__(self, hidden_dim, ff_dim, dropout_ratio): # ff_dim : FF layer의 자체 dimension
    super().__init__()

    self.fc1 = nn.Linear(hidden_dim, ff_dim)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(ff_dim, hidden_dim)

  def forward(self, out):
    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)

    return out
```

## Encoder

### Encoder Block(Layer)

```python
class EncoderBlock(nn.Module):
  def __init__(self, hidden_dim, n_heads, ff_dim, dropout_ratio, device):
    super().__init__()

    self.self_attention_layer = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
    self.self_attention_norm = nn.LayerNorm(hidden_dim)
    self.position_wise_ff_layer = PositionWiseFF(hidden_dim, ff_dim, dropout_ratio)
    self.position_wise_ff_norm = nn.LayerNorm(hidden_dim)
    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self, src, src_mask):
    # src: [batch_size, src_len, hidden_dim]
    # src_mask: [batch_size, src_len]
    
    # q,k,v = src, 필요시 src_mask 사용
    _src, a = self.self_attention_layer(src, src, src, src_mask)
    
    # residual connection : self.dropout(_src) 더함
    src = self.self_attention_norm(src + self.dropout(_src)) # 입력 src + attention
    _src = self.position_wise_ff_layer(src)
    src = self.position_wise_ff_norm(src + self.dropout(_src))

    return src # src: [batch_size, src_len, hidden_dim]
```

### Encoder

- Encoder Block 이 n_layers 만큼 이어진 구조

```python
class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_heads, ff_dim, dropout_ratio, device, n_layers, max_len=100):
    super().__init__()

    self.device = device

    # embedding 별도 선언 후 사용도 가능
    self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
    self.pos_embedding = nn.Embedding(max_len, hidden_dim)

    # n_layers 만큼 Encoder Block 반복
    self.layers = nn.ModuleList([EncoderBlock(hidden_dim, n_heads, ff_dim, dropout_ratio, device) for i in range(n_layers)])

    self.dropout = nn.Dropout(dropout_ratio)
    self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

  def forward(self, src, src_mask):
    # src: [batch_size, src_len]
    # src_mask: [batch_size, src_len]

    batch_size = src.shape[0] # batch 내의 문장 수
    src_len = src.shape[1] # batch 내 문장 중 가장 긴 것의 길이

    '''
    Positional Embedding
      pos: [batch_size, src_len]
      torch.arrange(0, src_len) : [src_len]
      unsqueeze(0) : [1, src_len]
      repeat(batch_size, 1) : [batch_size, src_len]
    '''
    pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

    # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
    src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)) # src: [batch_size, src_len, hidden_dim]
    
    for layer in self.layers:
      src = layer(src, src_mask)

    return src
```

## Decoder

### Decoder Block(Layer)

```python
class DecoderBlock(nn.Module):
  def __init__(self, hidden_dim, n_heads, ff_dim, dropout_ratio, device):
    super().__init__()

    self.self_attention_layer = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
    self.self_attention_norm = nn.LayerNorm(hidden_dim)
    self.encoder_attention = MultiHeadAttention(hidden_dim, n_heads, dropout_ratio, device)
    self.encoder_norm = nn.LayerNorm(hidden_dim)
    self.position_wise_ff_layer = PositionWiseFF(hidden_dim, ff_dim, dropout_ratio)
    self.position_wise_ff_norm = nn.LayerNorm(hidden_dim)

    self.dropout = nn.Dropout(dropout_ratio)

  def forward(self, tar, enc_src, tar_mask, src_mask):
    # trg: [batch_size, trg_len, hidden_dim]
    # enc_src: [batch_size, src_len, hidden_dim]
    # trg_mask: [batch_size, trg_len]
    # src_mask: [batch_size, src_len]
    
    # q,k,v = tar, 필요시 tar_mask 사용
    _tar, attention = self.self_attention_layer(tar, tar, tar, tar_mask) # trg: [batch_size, trg_len, hidden_dim]
    tar = self.self_attention_norm(tar + self.dropout(_tar))

    # Encoder Attention
    # query : tar
    # key, value : enc_src (encoder output)
    _tar, attention = self.encoder_attention(tar, enc_src, enc_src, src_mask)
    tar = self.encoder_norm(tar + self.dropout(_tar))

    _tar = self.position_wise_ff_layer(tar)
    tar = self.position_wise_ff_norm(tar + self.dropout(_tar))

    return tar, attention # tar: [batch_size, src_len, hidden_dim], attention : [batch_size, n_heads, hidden_dim, src_len]
```

### Decoder

- Decoder Block 이 n_layers 만큼 이어진 구조

```python
class Decoder(nn.Module):
  def __init__(self, output_dim, hidden_dim, n_heads, ff_dim, dropout_ratio, device, n_layers, max_len=100):
    super().__init__()

    self.device = device

    # embedding
    self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
    self.pos_embedding = nn.Embedding(max_len, hidden_dim)

    self.layers = nn.ModuleList([DecoderBlock(hidden_dim, n_heads, ff_dim, dropout_ratio, device) for i in range(n_layers)])

    self.fc_out = nn.Linear(hidden_dim, output_dim)

    self.dropout = nn.Dropout(dropout_ratio)
    self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

  def forward(self, tar, enc_src, tar_mask, src_mask):
    # trg: [batch_size, trg_len]
    # enc_src: [batch_size, src_len, hidden_dim]
    # trg_mask: [batch_size, trg_len]
    # src_mask: [batch_size, src_len]

    batch_size = tar.shape[0] # batch 내의 문장 수
    tar_len = tar.shape[1] # batch 내 문장 중 가장 긴 것의 길이

    pos = torch.arange(0, tar_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

    # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
    tar = self.dropout((self.tok_embedding(tar) * self.scale) + self.pos_embedding(pos)) # tar: [batch_size, tar_len, hidden_dim]
    
    for layer in self.layers:
      tar, attention = layer(tar, enc_src, tar_mask, src_mask)

    out = self.fc_out(tar) # out: [batch_size, trg_len, output_dim]

    return out
```

## Transformer

```python
class Transformer(nn.Module):
  def __init__(self, encoder, decoder, src_pad_idx, tar_pad_idx, device):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.tar_pad_idx = tar_pad_idx
    self.device = device

  def make_src_mask(self,src):
    # 소스 문장의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
    # src: [batch_size, src_len]

    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # src_mask: [batch_size, 1, 1, src_len]

    return src_mask

  def make_tar_mask(self, tar):
    # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    # tar: [batch_size, trg_len]

    tar_pad_mask = (tar != self.tar_pad_idx).unsqueeze(1).unsqueeze(2) # tar_pad_mask: [batch_size, 1, 1, trg_len]
    tar_len = tar.shape[1]
    tar_sub_mask = torch.tril(torch.ones((tar_len, tar_len), device = self.device)).bool() # tar_sub_mask: [tar_len, tar_len]

    tar_mask = tar_pad_mask & tar_sub_mask # tar_mask: [batch_size, 1, tar_len, tar_len]

    return tar_mask

  def forward(self, src, tar):
    # src: [batch_size, src_len]
    # trg: [batch_size, trg_len]

    src_mask = self.make_src_mask(src) # src_mask: [batch_size, 1, 1, src_len]
    tar_mask = self.make_tar_mask(tar) # tar_mask: [batch_size, 1, tar_len, tar_len]

    enc_src = self.encoder(src, src_mask)

    out = self.decoder(tar, enc_src, tar_mask, src_mask)

    return out
```

---

# Error Handling

### **SSL certificate error**

- Multi30k dataset 로드 과정에서 발생
- ssl certification 비활성화 및 data 직접 다운로드, 경로 생성

> **ref :** https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/issues/12, 
        https://github.com/pytorch/pytorch/issues/33288#issuecomment-1086779194
> 

```python
# error message
[ssl: certificate_verify_failed] certificate verify failed: 
hostname mismatch, certificate is not valid for 'www.quest.dcs.shef.ac.uk'. (_ssl.c:1007)]
```

**해결**

```python
# dataset load 전 실행
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

- [Dataset](https://github.com/zaidhassanch/PointerNetworks/tree/6ccd5ebad877c9fbc10ac3af10114b4a6097700b/data/multi30k) 다운로드 및 경로 `data/multi30k` 에 파일 로드

```python
# root='data' option 추가
train_dataset, valid_dataset, test_dataset 
	= Multi30k.splits(exts=('.de', '.en'), fields=(src, tgt), **root='data'**)
```

---

# Reference

- **torch.text - Field :** https://wikidocs.net/60314
- **spaCy :** https://ungodly-hour.tistory.com/37
- **BucketIterator :** https://torchtext.readthedocs.io/en/latest/data.html
- **Xavier Initialization :** [https://yngie-c.github.io/deep learning/2020/03/17/parameter_init/](https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/)
- **Residual Connection :** [https://velog.io/@stapers/논문-스터디-Week4-5-Attention-is-All-You-Need](https://velog.io/@stapers/%EB%85%BC%EB%AC%B8-%EC%8A%A4%ED%84%B0%EB%94%94-Week4-5-Attention-is-All-You-Need)
