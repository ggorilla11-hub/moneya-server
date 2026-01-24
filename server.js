const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ============================================
// RAG 데이터 로드 (1단계)
// ============================================
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

// ============================================
// RAG 검색 함수 (2단계)
// ============================================
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

// ============================================
// RAG 컨텍스트 생성 헬퍼 함수
// ============================================
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

// ============================================
// 시스템 프롬프트 생성 함수 (4단계: 3차 데이터 포함)
// ============================================
const createSystemPrompt = (userName, financialContext, budgetInfo, ragContext = '', designData = null) => {
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

  let prompt = `당신은 "머니야"입니다. ${name}님의 개인 AI 금융코치입니다.

## 호출 규칙 (최우선!)
- "${name}" 또는 "머니야"라고 부르면: "네, ${name}님!" 이것만 말하고 멈추세요
- 절대 추가 설명하지 마세요
- 그 다음 질문부터 정상 대화하세요

## 말투 규칙 (필수!)
- 반드시 존댓말을 사용하세요
- 공손하고 예의바르게 말하세요
- "~입니다", "~해요", "~하세요", "~할게요" 체를 사용하세요
- 절대 반말 금지: "~했어", "~할게", "~해봐" 사용하지 마세요

## 기본 규칙
- 한국어로만 대화하세요
- 이모지 절대 사용 금지
- 짧고 간결하게 말하세요 (최대 2-3문장)
- 항상 "${name}님"으로 호칭하세요

## 숫자 표기 규칙 (매우 중요!)

### 핵심 규칙
금액을 말할 때는 반드시 **한글**로만 말하세요!
숫자(1,2,3...)를 절대 사용하지 마세요!

### 올바른 응답 예시 (반드시 이 형식으로!)
- "오늘 남은 예산은 삼만사천구백육십사원입니다."
- "점심 예산으로 만오천원 사용하실 수 있어요."
- "이번 주 남은 예산은 십구만이천원이에요."
- "커피값 팔천원 정도는 괜찮아요."

### 한글 금액 표기 방법
- 35,207 → 삼만오천이백칠원
- 192,000 → 십구만이천원
- 66,667 → 육만육천육백육십칠원
- 15,000 → 만오천원
- 8,000 → 팔천원
- 1,500,000 → 백오십만원

### 절대 하지 말아야 할 것
- "34,964원입니다" ← 숫자 사용 금지!
- "34,964원(삼만사천구백육십사원)" ← 숫자+괄호 사용 금지!
- "15000원" ← 아라비아 숫자 절대 금지!
- 반드시 한글로만 금액을 표현하세요!

## ${name}님의 재무 현황

### 기본 정보 (1차 재무진단)
- 이름: ${name}
- 나이: ${age}세
- 월수입: ${monthlyIncome}만원

### 자산/부채 현황
- 총자산: ${totalAssets}만원
- 총부채: ${totalDebt}만원  
- 순자산: ${netAssets}만원
- 부자지수: ${wealthIndex}점
- 금융집 레벨: ${financialLevel}단계 (${houseName})

### 월 예산 배분 (2차 예산조정)
- 생활비: ${livingExpense.toLocaleString()}원
- 저축투자: ${savings.toLocaleString()}원
- 노후연금: ${pension.toLocaleString()}원
- 보장성보험: ${insurance.toLocaleString()}원
- 대출상환: ${loanPayment.toLocaleString()}원
- 잉여자금: ${surplus.toLocaleString()}원

### 오늘 예산
- 일일 예산: ${dailyBudget.toLocaleString()}원
- 오늘 지출: ${todaySpent.toLocaleString()}원
- 남은 예산: ${remainingBudget.toLocaleString()}원`;

  // 3차 금융집짓기 데이터 추가
  if (designData) {
    prompt += `\n\n### 금융집짓기 재무설계 (3차 데이터)`;
    
    // 은퇴설계
    if (designData.retire) {
      const r = designData.retire;
      prompt += `\n\n#### 은퇴설계
- 현재나이: ${r.currentAge || 0}세
- 은퇴예정: ${r.retireAge || 0}세
- 기대수명: ${r.lifeExpectancy || 0}세
- 월 필요생활비: ${(r.monthlyExpense || 0).toLocaleString()}원
- 국민연금 예상: ${(r.nationalPension || 0).toLocaleString()}원
- 개인연금 예상: ${(r.personalPension || 0).toLocaleString()}원`;
    }
    
    // 부채관리
    if (designData.debt) {
      const d = designData.debt;
      prompt += `\n\n#### 부채관리
- 월소득: ${(d.monthlyIncome || 0).toLocaleString()}원
- 주택담보대출 잔액: ${(d.mortgageBalance || 0).toLocaleString()}원 (금리 ${d.mortgageRate || 0}%)
- 주택담보대출 월상환: ${(d.mortgageMonthly || 0).toLocaleString()}원
- 신용대출 잔액: ${(d.creditBalance || 0).toLocaleString()}원 (금리 ${d.creditRate || 0}%)
- 신용대출 월상환: ${(d.creditMonthly || 0).toLocaleString()}원`;
    }
    
    // 저축설계
    if (designData.save) {
      const s = designData.save;
      prompt += `\n\n#### 저축설계
- 월소득: ${(s.monthlyIncome || 0).toLocaleString()}원
- 월저축액: ${(s.monthlySaving || 0).toLocaleString()}원
- 목표수익률: ${s.targetRate || 0}%`;
    }
    
    // 투자설계
    if (designData.invest) {
      const i = designData.invest;
      prompt += `\n\n#### 투자설계
- 현재나이: ${i.currentAge || 0}세
- 현재자산: ${(i.currentAssets || 0).toLocaleString()}원
- 월투자액: ${(i.monthlyInvestment || 0).toLocaleString()}원
- 기대수익률: ${i.expectedReturn || 0}%`;
    }
    
    // 세금설계
    if (designData.tax) {
      const t = designData.tax;
      prompt += `\n\n#### 세금설계
- 연소득: ${(t.annualIncome || 0).toLocaleString()}원
- 연금저축: ${(t.pensionSaving || 0).toLocaleString()}원
- IRP: ${(t.irpContribution || 0).toLocaleString()}원
- 주택청약: ${(t.housingSubscription || 0).toLocaleString()}원`;
    }
    
    // 부동산설계
    if (designData.estate) {
      const e = designData.estate;
      prompt += `\n\n#### 부동산설계
- 현재시세: ${(e.currentPrice || 0).toLocaleString()}원
- 대출잔액: ${(e.loanBalance || 0).toLocaleString()}원
- 월임대료: ${(e.monthlyRent || 0).toLocaleString()}원
- 보유기간: ${e.holdingYears || 0}년
- 예상상승률: ${e.expectedGrowth || 0}%`;
    }
    
    // 보험설계
    if (designData.insurance) {
      const ins = designData.insurance;
      prompt += `\n\n#### 보험설계
- 월보험료: ${(ins.monthlyPremium || 0).toLocaleString()}원
- 사망보장: ${(ins.deathCoverage || 0).toLocaleString()}원
- 질병보장: ${(ins.diseaseCoverage || 0).toLocaleString()}원
- 실손보험: ${ins.hasHealthInsurance ? '가입' : '미가입'}
- 연금보험: ${(ins.pensionInsurance || 0).toLocaleString()}원`;
    }
  }

  prompt += `\n\n## 대화 예시 (존댓말!)
- "오늘 남은 예산은 ${remainingBudget.toLocaleString()}원이에요. 무엇이 필요하세요?"
- "${name}님, 이번 달 저축 잘 하고 계시네요!"
- "커피 한 잔 정도는 괜찮으세요. 여유 있으시거든요."

${name}님의 든든한 금융 친구가 되어드릴게요!`;

  // RAG 컨텍스트가 있으면 추가
  if (ragContext) {
    prompt += `\n\n## 참고 자료 (오상열 CFP 지식)\n아래 내용을 참고하여 답변하되, 자연스럽게 녹여서 말하세요:\n${ragContext}`;
  }

  return prompt;
};

// Health check (버전 업데이트)
app.get('/', (req, res) => {
  res.json({ 
    status: 'AI머니야 서버 실행 중!', 
    version: '3.5',
    rag: { enabled: true, chunks: ragChunks.length }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// RAG 검색 테스트 API
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
    console.error('RAG Search Error:', error);
    res.json({ success: false, error: error.message });
  }
});

// 텍스트 채팅 API (4단계: 3차 데이터 포함)
app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo, designData } = req.body;
    
    // RAG 검색 및 컨텍스트 생성
    const ragContext = buildRAGContext(message);
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData);
    
    console.log('[Chat] RAG 검색 결과:', ragContext ? '있음' : '없음');
    console.log('[Chat] 3차 데이터:', designData ? '있음' : '없음');
    
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
    console.error('Chat API Error:', error);
    res.json({ success: false, message: '잠시 후 다시 시도해주세요.' });
  }
});

// TTS API (기존 그대로)
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'shimmer' } = req.body;
    const response = await openai.audio.speech.create({
      model: 'tts-1',
      voice: voice,
      input: text,
      response_format: 'mp3',
    });
    const buffer = Buffer.from(await response.arrayBuffer());
    const base64Audio = buffer.toString('base64');
    res.json({ success: true, audio: base64Audio });
  } catch (error) {
    console.error('TTS Error:', error);
    res.json({ success: false, error: 'TTS failed' });
  }
});

// HTTP 서버 시작
const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => {
  console.log(`AI머니야 서버 시작! 포트: ${PORT}`);
});

// ============================================
// WebSocket 서버 (4단계: 3차 데이터 포함)
// ============================================
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
  console.log('[Realtime] WebSocket 연결됨');
  
  let openaiWs = null;
  let userName = '고객';
  let financialContext = null;
  let budgetInfo = null;
  let designData = null;  // 3차 금융집짓기 데이터

  ws.on('message', (message) => {
    try {
      const msg = JSON.parse(message);

      if (msg.type === 'start_app') {
        console.log('[Realtime] 앱 시작 요청');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;
        budgetInfo = msg.budgetInfo || null;
        designData = msg.designData || null;  // 3차 데이터 수신
        
        console.log('[Realtime] 재무 정보 수신:', {
          name: financialContext?.name,
          age: financialContext?.age,
          wealthIndex: financialContext?.wealthIndex,
          hasDesignData: !!designData
        });

        openaiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {
          headers: {
            'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
            'OpenAI-Beta': 'realtime=v1'
          }
        });

        openaiWs.on('open', () => {
          console.log('[Realtime] OpenAI 연결됨!');
          // 초기 세션: 1차 + 2차 + 3차 데이터 포함
          const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, '', designData);
          
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

            // 사용자 음성 텍스트 수신 시 RAG 검색
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              const userText = event.transcript;
              console.log('사용자:', userText);
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));
              
              // RAG 검색 수행
              const ragContext = buildRAGContext(userText);
              
              if (ragContext) {
                console.log('[Realtime] RAG 검색 결과 있음, 세션 업데이트');
                
                // RAG 결과 + 3차 데이터를 포함한 새 프롬프트로 세션 업데이트
                const updatedPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData);
                
                openaiWs.send(JSON.stringify({
                  type: 'session.update',
                  session: {
                    instructions: updatedPrompt
                  }
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

      if (msg.type === 'audio' && openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify({
          type: 'input_audio_buffer.append',
          audio: msg.data
        }));
      }

      if (msg.type === 'stop') {
        console.log('[Realtime] 종료 요청');
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
