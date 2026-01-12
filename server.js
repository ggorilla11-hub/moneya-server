import Fastify from 'fastify';
import cors from '@fastify/cors';
import websocket from '@fastify/websocket';
import OpenAI from 'openai';
import WebSocket from 'ws';
import dotenv from 'dotenv';

dotenv.config();

const fastify = Fastify({ logger: true });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

await fastify.register(cors, { origin: true });
await fastify.register(websocket);

const createSystemPrompt = (userName, financialContext, budgetInfo) => {
  const name = financialContext?.name || userName || '사용자';
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
  const todaySaved = financialContext?.todaySaved || 0;

  return `당신은 "머니야"입니다. ${name}님의 개인 AI 금융코치입니다.

## 기본 규칙
- 한국어로만 대화하세요
- 이모지 절대 사용 금지
- 짧고 친근하게 말하세요 (최대 2-3문장)
- 반말로 친근하게 대화하세요
- 숫자는 읽기 쉽게 만원, 억 단위로 말하세요

## ${name}님의 재무 현황

### 기본 정보
- 이름: ${name}
- 나이: ${age}세
- 월수입: ${monthlyIncome}만원

### 자산/부채 현황
- 총자산: ${totalAssets}만원
- 총부채: ${totalDebt}만원  
- 순자산: ${netAssets}만원
- 부자지수: ${wealthIndex}점
- 금융집 레벨: ${financialLevel}단계 (${houseName})

### 월 예산 배분
- 생활비: ${livingExpense.toLocaleString()}원
- 저축투자: ${savings.toLocaleString()}원
- 노후연금: ${pension.toLocaleString()}원
- 보장성보험: ${insurance.toLocaleString()}원
- 대출상환: ${loanPayment.toLocaleString()}원
- 잉여자금: ${surplus.toLocaleString()}원

### 오늘 예산
- 일일 예산: ${dailyBudget.toLocaleString()}원
- 오늘 지출: ${todaySpent.toLocaleString()}원
- 남은 예산: ${remainingBudget.toLocaleString()}원

${name}님의 든든한 금융 친구가 되어줄게요!`;
};

fastify.get('/api/health', async () => {
  return { status: 'ok', timestamp: new Date().toISOString() };
});

fastify.post('/api/chat', async (request, reply) => {
  try {
    const { message, userName, financialContext, budgetInfo } = request.body;
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo);
    
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
    return { success: true, message: aiMessage };
  } catch (error) {
    console.error('Chat API Error:', error);
    return { success: false, message: '잠시 후 다시 시도해주세요.' };
  }
});

fastify.post('/api/tts', async (request, reply) => {
  try {
    const { text, voice = 'shimmer' } = request.body;
    const response = await openai.audio.speech.create({
      model: 'tts-1',
      voice: voice,
      input: text,
      response_format: 'mp3',
    });
    const buffer = Buffer.from(await response.arrayBuffer());
    const base64Audio = buffer.toString('base64');
    return { success: true, audio: base64Audio };
  } catch (error) {
    console.error('TTS Error:', error);
    return { success: false, error: 'TTS failed' };
  }
});

fastify.register(async function (fastify) {
  fastify.get('/ws/realtime', { websocket: true }, (connection, req) => {
    console.log('[Realtime] WebSocket connected');
    
    let openaiWs = null;
    let userName = '사용자';
    let financialContext = null;
    let budgetInfo = null;

    connection.on('message', async (message) => {
      try {
        const data = JSON.parse(message.toString());
        
        if (data.type === 'start_app') {
          console.log('[Realtime] App start request');
          userName = data.userName || '사용자';
          financialContext = data.financialContext || null;
          budgetInfo = data.budgetInfo || null;
          
          console.log('[Realtime] Financial info received:', {
            name: financialContext?.name,
            age: financialContext?.age,
            wealthIndex: financialContext?.wealthIndex,
            dailyBudget: budgetInfo?.dailyBudget
          });
          
          openaiWs = new WebSocket(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            { headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`, 'OpenAI-Beta': 'realtime=v1' } }
          );

          openaiWs.on('open', () => {
            console.log('[Realtime] OpenAI connected!');
            const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo);
            
            openaiWs.send(JSON.stringify({
              type: 'session.update',
              session: {
                modalities: ['text', 'audio'],
                instructions: systemPrompt,
                voice: 'shimmer',
                input_audio_format: 'pcm16',
                output_audio_format: 'pcm16',
                input_audio_transcription: { model: 'whisper-1' },
                turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500 },
                temperature: 0.7,
                max_response_output_tokens: 300,
              }
            }));
            
            connection.send(JSON.stringify({ type: 'session_started' }));
          });

          openaiWs.on('message', (msg) => {
            try {
              const event = JSON.parse(msg.toString());
              
              if (event.type === 'response.audio.delta' && event.delta) {
                connection.send(JSON.stringify({ type: 'audio', data: event.delta }));
              }
              if (event.type === 'response.audio_transcript.done' && event.transcript) {
                connection.send(JSON.stringify({ type: 'transcript', role: 'assistant', text: event.transcript }));
              }
              if (event.type === 'conversation.item.input_audio_transcription.completed' && event.transcript) {
                connection.send(JSON.stringify({ type: 'transcript', role: 'user', text: event.transcript }));
              }
              if (event.type === 'input_audio_buffer.speech_started') {
                connection.send(JSON.stringify({ type: 'interrupt' }));
              }
              if (event.type === 'error') {
                console.error('OpenAI error:', event.error);
                connection.send(JSON.stringify({ type: 'error', error: event.error?.message }));
              }
            } catch (e) {
              console.error('Message parse error:', e);
            }
          });

          openaiWs.on('error', (err) => {
            console.error('OpenAI WebSocket error:', err);
            connection.send(JSON.stringify({ type: 'error', error: 'Connection error' }));
          });

          openaiWs.on('close', () => {
            console.log('OpenAI WebSocket closed');
          });
        }
        
        if (data.type === 'audio' && openaiWs?.readyState === WebSocket.OPEN) {
          openaiWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: data.data }));
        }
        
        if (data.type === 'stop') {
          if (openaiWs) { openaiWs.close(); openaiWs = null; }
        }
      } catch (e) {
        console.error('Message handling error:', e);
      }
    });

    connection.on('close', () => {
      console.log('[Realtime] Client disconnected');
      if (openaiWs) { openaiWs.close(); openaiWs = null; }
    });
  });
});

const start = async () => {
  try {
    const port = process.env.PORT || 3001;
    await fastify.listen({ port: Number(port), host: '0.0.0.0' });
    console.log(`Server running on port ${port}`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
