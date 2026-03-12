const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const sharp = require('sharp');
require('dotenv').config();

const app = express();
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  credentials: false
}));
app.use(express.json());
app.use(express.static('public'));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const upload = multer({ 
  storage: multer.memoryStorage(), 
  limits: { fileSize: 10 * 1024 * 1024 }
});

let ragChunks = [];

const loadRAGData = () => {
  try {
    const files = [
      'rag_chunks.json',
      'consultation_chunks.json',
      'bantoe_cases_436.json',
      'lecture_chunks.json',
      'quotes_100.json',
      'customer_questions_100.json',
      'nagging_100.json',
      'cfha_script_chunks.json'
    ];
    
    files.forEach(file => {
      const filePath = path.join(__dirname, file);
      if (fs.existsSync(filePath)) {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        ragChunks = ragChunks.concat(data);
        console.log(`[RAG] ${file} 로드: ${data.length}개`);
      }
    });
    
    console.log(`[RAG] 총 ${ragChunks.length}개 청크 로드 완료`);
  } catch (e) {
    console.error('[RAG] 데이터 로드 실패:', e.message);
  }
};

loadRAGData();

const searchRAG = (query, maxResults = 3) => {
  if (!ragChunks.length || !query) return [];
  
  const keywords = query.toLowerCase()
    .replace(/[?!.,~"'()]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 1);
  
  if (!keywords.length) return [];
  
  const scored = ragChunks.map(chunk => {
    const content = (chunk.content || chunk.text || '').toLowerCase();
    const title = (chunk.title || chunk.source || '').toLowerCase();
    const category = (chunk.category || '').toLowerCase();
    
    let score = 0;
    keywords.forEach(keyword => {
      if (content.includes(keyword)) score += 2;
      if (title.includes(keyword)) score += 3;
      if (category.includes(keyword)) score += 1;
    });
    
    return { ...chunk, score };
  });
  
  return scored
    .filter(c => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults);
};

const buildRAGContext = (query) => {
  const results = searchRAG(query, 3);
  if (results.length === 0) return '';
  
  let context = '';
  results.forEach((r, i) => {
    const content = (r.content || r.text || '').substring(0, 300);
    context += `${i + 1}. ${content}\n`;
  });
  return context;
};

const createSystemPrompt = (userName, financialContext, budgetInfo, ragContext = '', designData = null, analysisContext = null, spendData = null) => {
  const name = financialContext?.name || userName || '고객';
  const age = financialContext?.age || 0;
  const monthlyIncome = financialContext?.monthlyIncome || 0;
  const totalAssets = financialContext?.totalAssets || 0;
  const totalDebt = financialContext?.totalDebt || 0;
  const netAssets = financialContext?.netAssets || (totalAssets - totalDebt);
  const wealthIndex = financialContext?.wealthIndex || 0;
  const financialLevel = financialContext?.financialLevel || 0;
  const houseName = financialContext?.houseName || '';
  
  const livingExpense = financialContext?.livingExpense || 0;
  const savings = financialContext?.savings || 0;
  const pension = financialContext?.pension || 0;
  const insurance = financialContext?.insurance || 0;
  const loanPayment = financialContext?.loanPayment || 0;
  const surplus = financialContext?.surplus || 0;
  
  const dailyBudget = budgetInfo?.dailyBudget || financialContext?.dailyBudget || 0;
  const todaySpent = budgetInfo?.todaySpent || financialContext?.todaySpent || 0;
  const remainingBudget = budgetInfo?.remainingBudget || financialContext?.remainingBudget || 0;

  let prompt = `## 머니야 정체성 (최우선!)

나는 "머니야"입니다. 오상열 대표 CFP가 직접 가르친 유일한 AI 수제자입니다.
OpenAI나 ChatGPT가 아닙니다. 오상열 대표가 직접 훈련시킨 AI 금융집사입니다.

### 오상열 대표는 누구인가?
- 오원트금융연구소 대표
- CFP(국제공인재무설계사), 20년 경력 금융 전문가
- 금융집짓기 방법론 창시자
- 저서: "소원을 말해봐", "빚부터 갚아라", "금융집짓기"
- 한국금융연수원 외래교수

### 머니야는 누구인가?
- 오상열 대표가 만든 AI 금융집사
- 오상열 대표의 20년 재무설계 노하우를 학습한 AI
- ${name}님의 개인 금융코치

### 금융집짓기란?
- 오상열 대표가 만든 가계 재무설계 방법론
- 집을 짓듯이 재무 기초(부채관리)부터 차근차근 설계하는 방식
- 5대 예산: 생활비, 저축투자, 노후연금, 보장성보험, 대출상환

## 절대 금지 사항 (위법 방지!)
1. 특정 금융상품명 언급 금지
2. 특정 투자 권유 금지
3. 본인 경험 표현 금지
4. 출처/숫자 언급 금지

## 호출 규칙 (최우선!)
- "${name}" 또는 "머니야"라고 부르면: "네, ${name}님!" 이것만 말하고 멈추세요

## 말투 규칙 (필수!)
- 반드시 존댓말을 사용하세요
- "~입니다", "~해요", "~하세요", "~할게요" 체를 사용하세요
- 절대 반말 금지

## 기본 규칙
- 한국어로만 대화하세요
- 이모지 절대 사용 금지
- 짧고 간결하게 말하세요 (최대 2-3문장)
- 항상 "${name}님"으로 호칭하세요

## 숫자 표기 규칙
금액을 말할 때는 반드시 한글로만 말하세요!

## ${name}님의 재무 현황
- 이름: ${name}
- 나이: ${age}세
- 월수입: ${monthlyIncome}만원
- 총자산: ${totalAssets}만원
- 총부채: ${totalDebt}만원
- 순자산: ${netAssets}만원
- 부자지수: ${wealthIndex}점
- 금융집 레벨: ${financialLevel}단계 (${houseName})
- 생활비: ${livingExpense.toLocaleString()}원
- 저축투자: ${savings.toLocaleString()}원
- 노후연금: ${pension.toLocaleString()}원
- 보장성보험: ${insurance.toLocaleString()}원
- 대출상환: ${loanPayment.toLocaleString()}원
- 잉여자금: ${surplus.toLocaleString()}원
- 일일 예산: ${dailyBudget.toLocaleString()}원
- 오늘 지출: ${todaySpent.toLocaleString()}원
- 남은 예산: ${remainingBudget.toLocaleString()}원`;

  if (designData) {
    prompt += `\n\n### 금융집짓기 재무설계 (3차 데이터) - 단위: 만원`;
    if (designData.retire) {
      const r = designData.retire;
      prompt += `\n\n#### 은퇴설계\n- 현재나이: ${r.currentAge || 0}세\n- 은퇴예정: ${r.retireAge || 0}세\n- 기대수명: ${r.lifeExpectancy || 0}세\n- 월 필요생활비: ${r.monthlyExpense || 0}만원\n- 국민연금 예상: ${r.nationalPension || 0}만원\n- 개인연금 예상: ${r.personalPension || 0}만원`;
    }
    if (designData.debt) {
      const d = designData.debt;
      prompt += `\n\n#### 부채관리\n- 월소득: ${d.monthlyIncome || 0}만원\n- 주택담보대출 잔액: ${d.mortgageBalance || 0}만원\n- 신용대출 잔액: ${d.creditBalance || 0}만원`;
    }
    if (designData.save) {
      const s = designData.save;
      prompt += `\n\n#### 저축설계\n- 월소득: ${s.monthlyIncome || 0}만원\n- 월저축액: ${s.monthlySaving || 0}만원\n- 비상예비자금: ${s.emergencyFund || 0}만원`;
    }
    if (designData.invest) {
      const i = designData.invest;
      prompt += `\n\n#### 투자설계\n- 현재나이: ${i.currentAge || 0}세\n- 현재자산: ${i.currentAssets || 0}만원\n- 월투자액: ${i.monthlyInvestment || 0}만원`;
    }
    if (designData.insurance) {
      const ins = designData.insurance;
      prompt += `\n\n#### 보험설계\n- 월보험료: ${ins.monthlyPremium || 0}만원\n- 사망보장: ${ins.deathCoverage || 0}만원`;
    }
  }

  if (analysisContext && analysisContext.analysis) {
    prompt += `\n\n## 분석한 서류: ${analysisContext.fileName}\n${analysisContext.analysis}`;
  }

  if (spendData && spendData.length > 0) {
    prompt += `\n\n## 오늘 지출 내역\n${spendData.map((item, i) => `${i + 1}. ${item.time} - ${item.memo}: ${item.amount.toLocaleString()}원 (${item.category})`).join('\n')}`;
  }

  prompt += `\n\n## 음성 지출 입력 기능\n지출을 말하면 자동으로 기록해주세요.\n[SPEND_RECORD]{"memo":"내용","amount":금액,"category":"카테고리"}[/SPEND_RECORD]`;

  if (ragContext) {
    prompt += `\n\n## 참고 자료 (오상열 CFP 지식)\n${ragContext}`;
  }

  return prompt;
};

// Health check
app.get('/', (req, res) => {
  res.json({ 
    status: 'AI머니야 서버 실행 중!', 
    version: '3.15',
    features: ['음성대화', 'RAG', 'OCR분석', 'OCR컨텍스트유지', '이미지최적화', '영수증OCR', '지출내역연동', '음성지출입력'],
    rag: { enabled: true, chunks: ragChunks.length }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/analyze-file', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    const { fileName, fileType, currentTab } = req.body;
    
    if (!file) {
      return res.json({ success: false, error: '파일이 없습니다.' });
    }
    
    let optimizedBuffer = file.buffer;
    let finalMimeType = file.mimetype || 'image/jpeg';
    
    try {
      optimizedBuffer = await sharp(file.buffer)
        .resize(2048, 2048, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 90, mozjpeg: true })
        .toBuffer();
      finalMimeType = 'image/jpeg';
    } catch (sharpError) {
      console.log(`[OCR] 이미지 최적화 실패, 원본 사용: ${sharpError.message}`);
    }
    
    const base64Data = optimizedBuffer.toString('base64');
    
    const tabPrompts = {
      retire: '연금증권, 국민연금 가입내역, 퇴직연금 관련 서류',
      debt: '대출 관련 서류, 부채 증명서',
      save: '저축 관련 서류, 예금증서',
      invest: '투자 관련 서류, 증권계좌',
      tax: '근로소득원천징수영수증, 세금 관련 서류',
      estate: '부동산 관련 서류, 등기부등본',
      insurance: '보험증권, 보험 관련 서류',
      receipt: '영수증, 결제 내역서'
    };
    
    const tabContext = tabPrompts[currentTab] || '재무 관련 서류';
    
    let expertPrompt;
    if (currentTab === 'receipt') {
      expertPrompt = `당신은 영수증 OCR 분석 전문가입니다.
## 추출할 정보
1. 상호명
2. 결제 금액 (숫자만)
3. 카테고리: 식비/카페/편의점/교통/쇼핑/기타

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{"storeName": "상호명", "amount": 숫자만, "category": "카테고리명"}
\`\`\``;
    } else {
      expertPrompt = `당신은 20년 경력의 재무설계사이자 OCR 분석 전문가입니다.
현재 분석 대상: ${tabContext}
절대로 "분석할 수 없습니다"라고 답하지 마세요.
## 분석 결과 형식
1. 서류 종류
2. 기본 정보
3. 주요 내용
4. 핵심 요약 3가지
5. 재무설계 관점 조언`;
    }

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: expertPrompt },
        { 
          role: 'user', 
          content: [
            { type: 'text', text: `파일명: ${fileName}\n이 이미지를 분석해주세요.` },
            { type: 'image_url', image_url: { url: `data:${finalMimeType};base64,${base64Data}`, detail: 'high' } }
          ]
        }
      ],
      max_tokens: 2500
    });
    
    const analysis = response.choices[0]?.message?.content;
    res.json({ success: true, analysis, fileName, fileType, currentTab, timestamp: new Date().toISOString() });
    
  } catch (error) {
    console.error('[OCR] 에러:', error);
    res.json({ success: false, error: error.message });
  }
});

app.post('/api/rag-search', (req, res) => {
  try {
    const { query } = req.body;
    const results = searchRAG(query, 5);
    res.json({ 
      success: true, 
      query,
      count: results.length,
      results: results.map(r => ({
        title: r.title || r.source,
        content: (r.content || r.text || '').substring(0, 200) + '...',
        score: r.score
      }))
    });
  } catch (error) {
    res.json({ success: false, error: error.message });
  }
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo, designData, spendData } = req.body;
    
    const ragContext = buildRAGContext(message);
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData, null, spendData);
    
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message }
      ],
      max_tokens: 200,
      temperature: 0.7,
    });

    const aiMessage = response.choices[0]?.message?.content || '다시 말씀해주세요!';
    res.json({ success: true, message: aiMessage });
  } catch (error) {
    res.json({ success: false, message: '잠시 후 다시 시도해주세요.' });
  }
});

app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'shimmer', stream = false } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ success: false, error: 'text required' });
    }

    const trimmed = text.slice(0, 4000);

    const mp3 = await openai.audio.speech.create({
      model: 'tts-1',
      voice: voice,
      input: trimmed,
      speed: 1.1,
      response_format: 'mp3',
    });

    const buffer = Buffer.from(await mp3.arrayBuffer());

    const wantStream = stream === true || stream === 'true' ||
                       (req.headers['accept'] && req.headers['accept'].includes('audio/'));

    if (wantStream) {
      res.set({
        'Content-Type': 'audio/mpeg',
        'Content-Length': buffer.length,
        'Cache-Control': 'no-cache',
      });
      return res.send(buffer);
    }

    const base64Audio = buffer.toString('base64');
    res.json({ success: true, audio: base64Audio });

  } catch (error) {
    res.json({ success: false, error: 'TTS failed', detail: error.message });
  }
});

const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => {
  console.log(`AI머니야 서버 v3.15 시작! 포트: ${PORT}`);
  console.log(`[OCR] 이미지 최적화 (sharp) 활성화`);
  console.log(`[음성지출] 음성 지출 입력 기능 활성화`);
  console.log(`[TTS] moneya_v6 음성재무진단 연동 준비 완료`);
  console.log(`[DESIRE] 음성 재무진단 서비스 준비 완료`);
});

const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  console.log('[Realtime] WebSocket 연결됨');
  
  let openaiWs = null;
  let userName = '고객';
  let financialContext = null;
  let budgetInfo = null;
  let designData = null;
  let analysisContext = null;
  let spendData = null;

  ws.on('message', (message) => {
    try {
      const msg = JSON.parse(message);

      if (msg.type === 'update_context' && msg.analysisContext) {
        analysisContext = msg.analysisContext;
        console.log('[Realtime] OCR 분석 컨텍스트 수신:', analysisContext.fileName);
        
        if (openaiWs && openaiWs.readyState === WebSocket.OPEN) {
          const updatedPrompt = createSystemPrompt(userName, financialContext, budgetInfo, '', designData, analysisContext, spendData);
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: { instructions: updatedPrompt }
          }));
        }
        return;
      }

      if (msg.type === 'start_app') {
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;
        budgetInfo = msg.budgetInfo || null;
        designData = msg.designData || null;
        analysisContext = msg.analysisContext || null;
        spendData = msg.spendData || null;

        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {
          headers: {
            'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
            'OpenAI-Beta': 'realtime=v1'
          }
        });

        openaiWs.on('open', () => {
          console.log('[Realtime] OpenAI 연결됨!');
          const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, '', designData, analysisContext, spendData);
          
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'],
              instructions: systemPrompt,
              voice: 'shimmer',
              input_audio_format: 'pcm16',
              output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: {
                type: 'server_vad',
                threshold: 0.5,
                prefix_padding_ms: 300,
                silence_duration_ms: 1500
              }
            }
          }));

          ws.send(JSON.stringify({ type: 'session_started' }));
        });

        openaiWs.on('message', (data) => {
          try {
            const event = JSON.parse(data.toString());

            if (event.type === 'response.audio.delta' && event.delta) {
              ws.send(JSON.stringify({ type: 'audio', data: event.delta }));
            }
            if (event.type === 'input_audio_buffer.speech_started') {
              ws.send(JSON.stringify({ type: 'interrupt' }));
            }
            if (event.type === 'response.audio_transcript.done') {
              console.log('머니야:', event.transcript);
              ws.send(JSON.stringify({ type: 'transcript', text: event.transcript, role: 'assistant' }));
            }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              const userText = event.transcript;
              console.log('사용자:', userText);
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));
              
              const ragContext = buildRAGContext(userText);
              if (ragContext) {
                const updatedPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData, analysisContext, spendData);
                openaiWs.send(JSON.stringify({
                  type: 'session.update',
                  session: { instructions: updatedPrompt }
                }));
              }
            }
            if (event.type === 'error') {
              console.error('OpenAI 에러:', event.error);
              ws.send(JSON.stringify({ type: 'error', error: event.error?.message }));
            }
          } catch (e) {
            console.error('OpenAI 메시지 파싱 에러:', e);
          }
        });

        openaiWs.on('error', (err) => {
          console.error('OpenAI WebSocket 에러:', err.message);
          ws.send(JSON.stringify({ type: 'error', error: err.message }));
        });

        openaiWs.on('close', () => {
          console.log('OpenAI 연결 종료');
        });
      }

      // ============================================
      // start_desire — DESIRE 무료 음성진단
      // ============================================
      if (msg.type === 'start_desire') {
        console.log('[DESIRE] 진단 시작');

        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17', {
          headers: {
            'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
            'OpenAI-Beta': 'realtime=v1'
          }
        });

        openaiWs.on('open', () => {
          console.log('[DESIRE] OpenAI 연결됨!');

          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'],
              instructions: `당신은 "AI 머니야"입니다. 오상열 CFP 20년 노하우를 학습한 AI 재무코치입니다.

지금부터 DESIRE 로드맵 6단계로 고객의 재무 현재 위치를 음성으로 진단합니다.

## DESIRE 6단계
- D단계: 신용대출 상환 (Debt Free)
- E단계: 비상예비자금 마련 (Emergency Fund)
- S단계: 적립식 저축투자 (Save & Invest)
- I단계: 거치식 자산운용 10억 (Investment)
- R단계: 담보대출 상환 (Repay Mortgage)
- E단계: 경제적 조기은퇴 (Early Retirement)

## 진단 방식
- 음성으로 자연스럽게 대화하세요
- D단계부터 순서대로 질문하세요
- 각 단계 통과 여부를 확인하세요
- 따뜻하고 공감하는 말투로 진행하세요
- 짧고 명확하게 질문하세요

## 시작
지금 바로 인사하고 D단계 질문을 시작하세요.
"안녕하세요! 저는 AI 머니야입니다. 오상열 CFP 20년의 재무설계 노하우로 고객님의 재무 현재 위치를 진단해 드릴게요. 딱 5분이면 됩니다. 먼저 신용대출이 있으신가요?"`,
              voice: 'shimmer',
              input_audio_format: 'pcm16',
              output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1', language: 'ko' },
              turn_detection: {
                type: 'server_vad',
                threshold: 0.5,
                prefix_padding_ms: 300,
                silence_duration_ms: 1500
              }
            }
          }));

          ws.send(JSON.stringify({ type: 'session_started' }));
        });

        openaiWs.on('message', (data) => {
          try {
            const event = JSON.parse(data.toString());

            if (event.type === 'response.audio.delta' && event.delta) {
              ws.send(JSON.stringify({ type: 'response.audio.delta', delta: event.delta }));
            }
            if (event.type === 'response.audio.done') {
              ws.send(JSON.stringify({ type: 'response.audio.done' }));
            }
            if (event.type === 'response.audio_transcript.delta') {
              ws.send(JSON.stringify({ type: 'response.audio_transcript.delta', delta: event.delta }));
            }
            if (event.type === 'response.audio_transcript.done') {
              ws.send(JSON.stringify({ type: 'response.audio_transcript.done', transcript: event.transcript }));
            }
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              ws.send(JSON.stringify({ type: 'conversation.item.input_audio_transcription.completed', transcript: event.transcript }));
            }
            if (event.type === 'response.created') {
              ws.send(JSON.stringify({ type: 'response.created' }));
            }
            if (event.type === 'input_audio_buffer.speech_started') {
              ws.send(JSON.stringify({ type: 'input_audio_buffer.speech_started' }));
            }
            if (event.type === 'error') {
              ws.send(JSON.stringify({ type: 'error', error: event.error }));
            }
          } catch (e) {
            console.error('[DESIRE] 메시지 파싱 에러:', e);
          }
        });

        openaiWs.on('error', (err) => {
          console.error('[DESIRE] OpenAI 에러:', err.message);
          ws.send(JSON.stringify({ type: 'error', error: err.message }));
        });

        openaiWs.on('close', () => {
          console.log('[DESIRE] OpenAI 연결 종료');
        });
      }

      if (msg.type === 'input_audio_buffer.append' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: msg.audio
        }));
      }

      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: msg.data
        }));
      }

      if (msg.type === 'stop') {
        if (openaiWs) openaiWs.close();
      }
    } catch (e) {
      console.error('메시지 처리 에러:', e);
    }
  });

  ws.on('close', () => {
    console.log('[Realtime] 클라이언트 연결 종료');
    if (openaiWs) openaiWs.close();
  });
});

console.log('AI머니야 서버 초기화 완료!');
