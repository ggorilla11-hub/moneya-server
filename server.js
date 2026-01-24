/**
 * AI머니야 서버 v7.0 - RAG 2,766개 청크 통합
 * 파트1: 설정 + RAG 시스템
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { WebSocketServer, WebSocket } = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '50mb' }));

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PORT = process.env.PORT || 10000;

// ========================================
// RAG 시스템 - 2,766개 청크
// ========================================

let ragChunks = [];

function loadRAGData() {
  try {
    const ragPath = path.join(__dirname, 'rag_chunks.json');
    if (fs.existsSync(ragPath)) {
      const data = fs.readFileSync(ragPath, 'utf8');
      ragChunks = JSON.parse(data);
      console.log(`✅ RAG 로드: ${ragChunks.length}개 청크`);
      
      const bookCounts = {};
      ragChunks.forEach(c => {
        const book = c.book || '기타';
        bookCounts[book] = (bookCounts[book] || 0) + 1;
      });
      Object.entries(bookCounts).forEach(([book, count]) => {
        console.log(`   📚 ${book}: ${count}개`);
      });
      return true;
    }
    console.log('⚠️ RAG 파일 없음');
    return false;
  } catch (error) {
    console.error('❌ RAG 로드 실패:', error.message);
    return false;
  }
}

function searchRAG(query, maxResults = 5) {
  if (!ragChunks || ragChunks.length === 0) return [];
  
  const keywords = query.toLowerCase()
    .replace(/[?!.,。、]/g, '')
    .split(/\s+/)
    .filter(k => k.length > 1);
  
  if (keywords.length === 0) return [];
  
  const scored = ragChunks.map(chunk => {
    const content = (chunk.content || '').toLowerCase();
    const book = (chunk.book || '').toLowerCase();
    let score = 0;
    
    keywords.forEach(keyword => {
      const matches = (content.match(new RegExp(keyword, 'g')) || []).length;
      score += matches * 2;
      if (book.includes(keyword)) score += 5;
      if (chunk.type === 'quote' && content.includes(keyword)) score += 3;
      if (chunk.type === 'consultation' && content.includes(keyword)) score += 4;
    });
    
    return { ...chunk, score };
  });
  
  return scored
    .filter(c => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults)
    .map(({ score, ...chunk }) => chunk);
}

function formatRAGContext(results) {
  if (!results || results.length === 0) return '';
  
  let context = '\n\n[참고자료]\n';
  results.forEach((chunk) => {
    const source = chunk.book || '참고자료';
    context += `\n【${source}】\n${chunk.content.substring(0, 500)}...\n`;
  });
  context += '\n[위 자료를 자연스럽게 활용하여 답변하세요]\n';
  return context;
}

loadRAGData();
// ========================================
// 시스템 프롬프트 생성 (고객정보 연결)
// ========================================

function createSystemPrompt(userName, financialContext, budgetInfo, designData) {
  const name = financialContext?.name || userName || '고객';
  const age = financialContext?.age || 0;
  const monthlyIncome = financialContext?.monthlyIncome || 0;
  const totalAssets = financialContext?.totalAssets || 0;
  const totalDebt = financialContext?.totalDebt || 0;
  const wealthIndex = financialContext?.wealthIndex || 0;
  const financialLevel = financialContext?.financialLevel || 0;
  const houseName = financialContext?.houseName || '';
  
  const livingExpense = financialContext?.livingExpense || 0;
  const savings = financialContext?.savings || 0;
  const pension = financialContext?.pension || 0;
  const insurance = financialContext?.insurance || 0;
  const loanPayment = financialContext?.loanPayment || 0;
  
  const dailyBudget = budgetInfo?.dailyBudget || 0;
  const todaySpent = budgetInfo?.todaySpent || 0;
  const remainingBudget = budgetInfo?.remainingBudget || 0;
  
  const job = designData?.job || '';
  const housingType = designData?.housingType || '';
  const financialGoal = designData?.financialGoal || '';
  const desireLevel = designData?.desireLevel || '';

  return `당신은 "머니야"입니다. 오상열 CFP가 20년 경력으로 직접 가르친 유일한 AI 금융코치입니다.

## 정체성
- 이름: 머니야 (AI 금융집사)
- 스승: 오상열 CFP (재무설계 전문가, 저서 3권, 17년간 반퇴시대 칼럼니스트)
- 학습: 2,766개의 실제 상담사례, 강의, 책을 학습한 전문 AI

## ${name}님 재무현황

### 1차 재무진단
- 나이: ${age}세 / 월수입: ${monthlyIncome.toLocaleString()}만원
- 총자산: ${totalAssets.toLocaleString()}만원 / 총부채: ${totalDebt.toLocaleString()}만원
- 부자지수: ${wealthIndex}% / 금융집: ${financialLevel}단계 ${houseName}

### 2차 재무분석
- 생활비: ${livingExpense.toLocaleString()}원 / 저축: ${savings.toLocaleString()}원
- 연금: ${pension.toLocaleString()}원 / 보험: ${insurance.toLocaleString()}원 / 대출상환: ${loanPayment.toLocaleString()}원

### 오늘 예산
- 일일: ${dailyBudget.toLocaleString()}원 / 지출: ${todaySpent.toLocaleString()}원 / 남은: ${remainingBudget.toLocaleString()}원

${job ? `### 3차 금융집짓기\n- 직업: ${job} / 주거: ${housingType}\n- 목표: ${financialGoal} / DESIRE: ${desireLevel}` : ''}

## 대화규칙
- 반드시 존댓말 ("~요", "~습니다")
- "${name}님" 호출시: "네, ${name}님!" 만 답하고 멈춤
- 답변은 간결하게 (3-4문장)

## 금융집짓기® 원칙
1. 5대예산: 저축(20-50%), 주거(25%), 보험연금(10%), 생활비(20-60%), 대출(10%)
2. 저축은 근육, 대출은 암덩어리
3. 수입 - 저축 = 지출

## 금지사항
- 반말 금지 / 특정 금융상품 브랜드 언급 금지 / 투자 권유 금지`;
}

// ========================================
// REST API
// ========================================

app.get('/', (req, res) => {
  res.json({
    status: 'AI머니야 서버 v7.0',
    rag: { enabled: ragChunks.length > 0, chunks: ragChunks.length }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/rag-search', (req, res) => {
  const { query, maxResults = 5 } = req.body;
  if (!query) return res.status(400).json({ error: '검색어 필요' });
  const results = searchRAG(query, maxResults);
  res.json({ query, count: results.length, results });
});

app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName = '고객', financialContext, budgetInfo, designData } = req.body;
    if (!message) return res.status(400).json({ error: '메시지 필요' });
    
    const ragResults = searchRAG(message, 3);
    const ragContext = formatRAGContext(ragResults);
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, designData);
    
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${OPENAI_API_KEY}` },
      body: JSON.stringify({
        model: 'gpt-4o',
        messages: [
          { role: 'system', content: systemPrompt + ragContext },
          { role: 'user', content: message }
        ],
        max_tokens: 1000, temperature: 0.7
      })
    });
    
    const data = await response.json();
    if (data.error) throw new Error(data.error.message);
    
    res.json({
      success: true,
      message: data.choices[0].message.content,
      ragUsed: ragResults.length > 0
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ========================================
// WebSocket (Realtime API)
// ========================================

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

wss.on('connection', (ws, req) => {
  let openaiWs = null;
  let financialContext = null;
  let budgetInfo = null;
  let designData = null;
  let userName = '고객';
  
  ws.on('message', async (data) => {
    try {
      const message = JSON.parse(data.toString());
      
      if (message.type === 'start_app') {
        financialContext = message.financialContext || null;
        budgetInfo = message.budgetInfo || null;
        designData = message.designData || null;
        userName = message.userName || financialContext?.name || '고객';
        
        const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, designData);
        
        openaiWs = new WebSocket(
          'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
          { headers: { 'Authorization': `Bearer ${OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' } }
        );
        
        openaiWs.on('open', () => {
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              modalities: ['text', 'audio'],
              instructions: systemPrompt,
              voice: 'shimmer',
              input_audio_format: 'pcm16',
              output_audio_format: 'pcm16',
              input_audio_transcription: { model: 'whisper-1' },
              turn_detection: { type: 'server_vad', threshold: 0.5, silence_duration_ms: 800 }
            }
          }));
          ws.send(JSON.stringify({ type: 'session_started', message: `네, ${userName}님!` }));
        });
        
        openaiWs.on('message', (d) => { if (ws.readyState === WebSocket.OPEN) ws.send(d); });
        openaiWs.on('error', (e) => ws.send(JSON.stringify({ type: 'error', message: e.message })));
        openaiWs.on('close', () => {});
        
      } else if (openaiWs && openaiWs.readyState === WebSocket.OPEN) {
        openaiWs.send(JSON.stringify(message));
      }
    } catch (error) { console.error(error); }
  });
  
  ws.on('close', () => { if (openaiWs) openaiWs.close(); });
});

server.listen(PORT, () => {
  console.log(`✅ AI머니야 서버 v7.0 시작 - 포트 ${PORT}`);
  console.log(`📊 RAG: ${ragChunks.length}개 청크`);
});
