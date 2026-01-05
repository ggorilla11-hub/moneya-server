const express = require('express');
const cors = require('cors');
const Anthropic = require('@anthropic-ai/sdk').default;
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 10000;

// 미들웨어
app.use(cors());
app.use(express.json());

// Claude 클라이언트
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// AI머니야 시스템 프롬프트
const MONEYA_SYSTEM_PROMPT = `당신은 AI머니야입니다. 오상열 CFP의 20년 재무설계 노하우를 학습한 AI 금융코치입니다.

## 역할
- 친근하고 유머러스한 금융코치
- 상품 판매 없이 순수하게 고객의 재무 건강을 돕는 조력자
- "이거 사도 돼?"라는 질문에 현명한 조언을 주는 친구

## 대화 스타일 (4단계 원칙)
1. 좋은 것 먼저 (긍정): "대표님, 이번 달 저축률이 올랐어요!"
2. 살짝 아쉬운 것 (현실): "다만, 식비가 예산보다 15% 초과했어요"
3. 근거 제시 (시뮬레이션): "지금 페이스라면 3년 후 순자산 2억 달성 가능해요"
4. 마무리 (선택권 부여): "어떻게 하실까요?"

## 핵심 원칙
1. 절대 특정 금융상품을 추천하지 않습니다
2. 투자 수익을 보장하는 발언을 하지 않습니다
3. 10년 후가 아닌 3년 이내 시뮬레이션을 제시합니다
4. 응답은 간결하게 2-3문장으로 합니다

## 금지 표현
- "결정은 대표님 몫이에요" → "어떻게 하실까요?"로 대체
- 책임 회피 표현 금지, 손 잡고 안내하는 표현 사용`;

// 서버 상태 확인
app.get('/', (req, res) => {
  res.json({ 
    status: 'OK', 
    service: 'AI머니야 백엔드',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
});

// AI 채팅 엔드포인트
app.post('/api/chat', async (req, res) => {
  try {
    const { message, context = [] } = req.body;

    const messages = [
      ...context,
      { role: 'user', content: message }
    ];

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1024,
      system: MONEYA_SYSTEM_PROMPT,
      messages: messages
    });

    res.json({ 
      success: true, 
      response: response.content[0].text,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Chat Error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// 서버 시작
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════╗
║         AI머니야 백엔드 서버            ║
║                                        ║
║  PORT: ${PORT}                           ║
║  STATUS: Running                       ║
║                                        ║
║  오원트금융연구소 | 오상열 CFP          ║
╚════════════════════════════════════════╝
  `);
});