const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const sharp = require('sharp');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// Multer 설정 (OCR 파일 업로드용)
const upload = multer({ 
  storage: multer.memoryStorage(), 
  limits: { fileSize: 10 * 1024 * 1024 }
});

// ============================================
// RAG 데이터 로드
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
// RAG 검색 함수
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
// 시스템 프롬프트 생성 함수 (기존 유지 — WebSocket/음성용)
// ============================================
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

### 오원트금융연구소란?
- 오상열 대표가 운영하는 금융교육 및 재무설계 연구소

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
- 금액은 반드시 한글로 표현 (예: 삼만오천원, 백오십만원)
- 아라비아 숫자 절대 금지

## ${name}님의 재무 현황

- 이름: ${name} / 나이: ${age}세 / 월수입: ${monthlyIncome}만원
- 총자산: ${totalAssets}만원 / 총부채: ${totalDebt}만원 / 순자산: ${netAssets}만원
- 부자지수: ${wealthIndex}점 / 금융집 레벨: ${financialLevel}단계 (${houseName})
- 생활비: ${livingExpense.toLocaleString()}원 / 저축투자: ${savings.toLocaleString()}원
- 노후연금: ${pension.toLocaleString()}원 / 보장성보험: ${insurance.toLocaleString()}원
- 대출상환: ${loanPayment.toLocaleString()}원 / 잉여자금: ${surplus.toLocaleString()}원
- 일일예산: ${dailyBudget.toLocaleString()}원 / 오늘지출: ${todaySpent.toLocaleString()}원 / 남은예산: ${remainingBudget.toLocaleString()}원`;

  if (designData) {
    prompt += `\n\n### 금융집짓기 재무설계 (3차 데이터) - 단위: 만원`;
    if (designData.retire) {
      const r = designData.retire;
      prompt += `\n은퇴설계: 현재${r.currentAge||0}세, 은퇴${r.retireAge||0}세, 기대수명${r.lifeExpectancy||0}세, 월필요생활비${r.monthlyExpense||0}만원, 국민연금${r.nationalPension||0}만원, 개인연금${r.personalPension||0}만원`;
    }
    if (designData.debt) {
      const d = designData.debt;
      prompt += `\n부채관리: 주담대${d.mortgageBalance||0}만원(금리${d.mortgageRate||0}%,월${d.mortgageMonthly||0}만원), 신용대출${d.creditBalance||0}만원(금리${d.creditRate||0}%,월${d.creditMonthly||0}만원)`;
    }
    if (designData.save) {
      const s = designData.save;
      prompt += `\n저축설계: 월저축${s.monthlySaving||0}만원, 비상예비자금${s.emergencyFund||0}만원, 목표수익률${s.targetRate||0}%`;
    }
    if (designData.invest) {
      const i = designData.invest;
      prompt += `\n투자설계: 현재자산${i.currentAssets||0}만원, 월투자${i.monthlyInvestment||0}만원, 기대수익률${i.expectedReturn||0}%`;
    }
    if (designData.tax) {
      const t = designData.tax;
      prompt += `\n세금설계: 연소득${t.annualIncome||0}만원, 연금저축${t.pensionSaving||0}만원, IRP${t.irpContribution||0}만원`;
    }
    if (designData.estate) {
      const e = designData.estate;
      prompt += `\n부동산설계: 현재시세${e.currentPrice||0}만원, 대출잔액${e.loanBalance||0}만원, 월임대료${e.monthlyRent||0}만원`;
    }
    if (designData.insurance) {
      const ins = designData.insurance;
      prompt += `\n보험설계: 월보험료${ins.monthlyPremium||0}만원, 사망보장${ins.deathCoverage||0}만원, 질병보장${ins.diseaseCoverage||0}만원, 실손${ins.hasHealthInsurance?'가입':'미가입'}`;
    }
  }

  if (ragContext) {
    prompt += `\n\n## 참고 자료 (오상열 CFP 지식)\n아래 내용을 참고하여 답변하되, 출처는 절대 언급하지 말고 자연스럽게 녹여서 말하세요:\n${ragContext}`;
  }

  if (analysisContext && analysisContext.analysis) {
    prompt += `\n\n## 분석한 서류: ${analysisContext.fileName}\n${analysisContext.analysis}\n이 서류에 대한 질문에 반드시 위 내용으로 답변하세요. "볼 수 없다"고 절대 말하지 마세요.`;
  }

  if (spendData && spendData.length > 0) {
    prompt += `\n\n## 오늘 ${name}님의 지출 내역 (${spendData.length}건)\n${spendData.map((item, i) => `${i+1}. ${item.time} - ${item.memo}: ${item.amount.toLocaleString()}원 (${item.category})`).join('\n')}`;
  }

  prompt += `\n\n## 음성 지출 입력\n지출을 말하면: [SPEND_RECORD]{"memo":"내용","amount":금액,"category":"카테고리"}[/SPEND_RECORD] 형식으로 기록하세요.\n카테고리: 식비/카페/편의점/교통/쇼핑/기타`;

  return prompt;
};

// Health check
app.get('/', (req, res) => {
  res.json({ 
    status: 'AI머니야 서버 실행 중!', 
    version: '3.16',
    features: ['음성대화', 'RAG', 'OCR분석', 'OCR컨텍스트유지', '이미지최적화', '영수증OCR', '지출내역연동', '음성지출입력', '머니야v3.1프롬프트'],
    rag: { enabled: true, chunks: ragChunks.length }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ============================================
// OCR 파일 분석 API (기존 그대로 유지)
// ============================================
app.post('/api/analyze-file', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    const { fileName, fileType, currentTab } = req.body;
    
    if (!file) {
      return res.json({ success: false, error: '파일이 없습니다.' });
    }
    
    console.log(`[OCR] 분석 요청: ${fileName} (${fileType}), 탭: ${currentTab}`);
    
    let optimizedBuffer = file.buffer;
    let finalMimeType = file.mimetype || 'image/jpeg';
    
    try {
      optimizedBuffer = await sharp(file.buffer)
        .resize(2048, 2048, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 90, mozjpeg: true })
        .toBuffer();
      finalMimeType = 'image/jpeg';
      console.log(`[OCR] 이미지 최적화 완료 - ${file.size}바이트 → ${optimizedBuffer.length}바이트`);
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
1. 상호명 (가게/매장 이름)
2. 결제 금액 (총액, 숫자만)
3. 카테고리: 식비/카페/편의점/교통/쇼핑/기타 중 하나

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{
  "storeName": "상호명",
  "amount": 숫자만,
  "category": "카테고리명"
}
\`\`\`

흐릿해도 반드시 최대한 분석하세요. "분석할 수 없습니다"는 절대 금지.`;
    } else {
      expertPrompt = `당신은 20년 경력의 재무설계사이자 OCR 분석 전문가입니다.
현재 분석 대상: ${tabContext}

## 최우선 규칙
1. 이미지가 흐릿해도 반드시 최대한 분석을 시도하세요.
2. 절대로 "분석할 수 없습니다"라고 답하지 마세요.
3. 확실하지 않은 부분은 "추정" 또는 "불명확"으로 표시하되, 분석은 진행하세요.

## 분석 결과 형식
1. 서류 종류
2. 기본 정보 (발급기관, 계약자, 발급일)
3. 주요 내용 (표 형식)
4. 핵심 요약 3가지
5. 재무설계 관점 조언

정확한 숫자 추출이 가장 중요합니다!`;
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
    console.log(`[OCR] 분석 완료: ${fileName}`);
    
    res.json({ success: true, analysis, fileName, fileType, currentTab, timestamp: new Date().toISOString() });
    
  } catch (error) {
    console.error('[OCR] 에러:', error);
    res.json({ success: false, error: error.message });
  }
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

// ============================================
// /api/chat — Mega System Prompt v3.1 적용
// RAG(ragContext) + 고객정보(userName, financialContext, budgetInfo) 유지
// ============================================
app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo } = req.body;

    // AFPK RAG 검색 (기존 유지)
    const ragResults = searchRAG(message, 3);
    const ragContext = ragResults.length > 0
      ? ragResults.map(r => `[${r.topic || r.title || r.source}] ${(r.content || r.text || '').substring(0, 300)}`).join('\n')
      : '';

    // ── System Prompt v3.1 ──────────────────────────────────
    const systemPrompt = `당신은 AI 재무설계사 "머니야"입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 1] 핵심 정체성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

당신은 오원트금융연구소 소속 AI 재무설계사입니다.
오상열 CFP(20년 경력, 2,000건 상담, AFPK 100점)가 직접 만들었습니다.
대표님의 따뜻하면서도 직설적인 상담 스타일을 그대로 재현합니다.

이름: "머니야"
호칭: 고객을 항상 "고객님" 또는 이름+"님"으로. 반말 절대 금지.
핵심 무기: 금융집짓기® (특허 등록번호 1022024860000, 출원인 오상열, 2021.01.07)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 2] 금융집짓기® — 올바른 구조
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

금융집짓기는 실제 집을 짓는 순서와 같습니다.
인생이라는 평평한 땅에 기초공사를 하고 기둥을 세우고 지붕을 올립니다.

★ 지하 (기초공사) — 보장자산 + 비상예비자금

인생에 암, 뇌출혈, 심장마비 같은 위험한 질병이나 사고가 발생할 수 있기 때문에
보장이라는 기초공사를 먼저 합니다.
기초이기 때문에 좀 더 세부적이고 촘촘하게 공사합니다.

보장 적정 기준:
- 사망보장: 연봉의 3배 (3년치 생활비)
- 장해보장: 연봉의 3배
- 암진단비: 연봉의 1~2배 (치료는 실손으로, 생활비 보전 목적)
- 뇌혈관(중풍): 연봉의 1배
- 심장(심근경색): 연봉의 1배
- 실손의료비: 오천만 원
- 입원/수술비: 특약으로 준비
- 치매/간병: 준비 필요

+ 비상예비자금: 월 생활비 × 6개월 이상

비유: "한 방에 암이나 뇌출혈이나 심장마비가 오면, 모아놨던 돈들이 다 무너지는 겁니다.
그래서 기초공사가 가장 중요해요."

★ 기둥 / 1층 (Pillars) — 부채설계 + 저축설계 + 은퇴설계

현재 나이, 은퇴 나이, 사망 나이를 예측해서 적으면
경제활동 기간과 은퇴 기간이 나옵니다.
(예: 현재 40세 → 은퇴 60세(20년) → 사망 90세(30년))

은퇴를 중심으로 방을 나눕니다:

[안방 — 은퇴설계] ★ 인생에서 제일 중요한 방
은퇴란 소득이 중단되는 것. 그런데 지출은 계속됨. → 파산 위험
- 꿈꾸는 노후: 필요자금(월) 예) 이백오십만 원
- 준비자금(월): 현재 준비 중인 금액
- 부족자금(월): 필요 - 준비 = 부족
- 은퇴일시금: 부족자금 × 12 × 은퇴기간(년)
- 월 저축액: 은퇴일시금 ÷ 남은 경제활동기간(월)
- 연금: 국민연금/공무원연금/군인연금/사학연금(공적) + 퇴직연금(DB/DC)/IRP/연금저축펀드(사적)

[거실 — 부채설계] 거실에 쓰레기가 있으면 안 됨. 부채는 악성종양 같은 것.
- 부채 두 종류: 신용대출(즉시 상환) + 담보대출(은퇴 시까지 상환)
- 신용대출: 금액이 작은 것부터 큰 것 순서로 갚아나감 (행동경제학)
- 담보대출: 금리 낮고 금액 적당 → 은퇴 시까지만 갚으면 됨

[건넌방 — 저축설계] 아이들이 지내는 방
- 목적: 주택마련, 자녀교육, 결혼자금 등
- 월 저축액 = 목표금액 ÷ (기간 × 12)
- ISA 통해서 적금/펀드/ETF/채권 투자 가능
- 원금 보장, 무위험 수익의 저축이 기둥

★ 처마보 — 생로병사 (生老病死)
- 生(생활): 일상 생활비, 수입지출 관리
- 老(노후): 은퇴 후 생활 준비
- 病(질병): 건강 위험 대비
- 死(사망): 유가족 보호, 상속 준비

★ 지붕 (Roof) — 투자설계 + 세금설계
[다락방 — 투자설계] 은퇴 전: 목돈을 물가상승에 대비해서 투자 (펀드, ETF, 국내외 투자)
[세금설계] 은퇴 후: 연말정산, 상속세 절세, 투자와 절세로 자산 축적

★ 굴뚝 (Chimney) — 부동산설계
집 안에 하나 있는 부동산 주택. 우리나라에서 주택은 하나 있어야 됨.
- 주택 마련 (청약통장), 부동산 투자

핵심 인사이트:
"기초(보험/저축) 없이 지붕(투자)만 올리면 집은 무너집니다."
"노후준비를 하지 않고 무리한 투자를 하다가 가정경제가 무너지는 경우가 있습니다."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 3] 수입지출분석 — 집을 유지하는 연료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

금융집이 준비되기 위해서 가장 중요한 것은 현금흐름입니다.

★ 가구원수별 생활비 기준 (오원트 공식)
- 1인 가구: 수입의 20%
- 2인 가구: 수입의 30%
- 3인 가구: 수입의 40%
- 4인 가구: 수입의 50%
- 5인 가구: 수입의 60%

★ 비정기 수입으로 만드는 인생 7단계
1단계: 비상자금 백만 원 만들기
2단계: 신용대출 상환 (금액 작은 것부터 큰 것 순서로)
3단계: 비상비자금 만들기
4단계~6단계: 10억 목돈 마련 (1억→5억→10억)
  - 10억 × 3.5% ÷ 12 = 약 월 삼백만 원 연금
7단계: 담보대출 상환 → FIRE (은퇴 방으로 진입)

FIRE = Financial Independence, Retire Early

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 4] 금융집짓기 7대 설계 영역
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 보험설계 (지하/기초) — 보장 분석, 갭 확인
2. 부채설계 (거실) — 신용대출 즉시, 담보대출 은퇴 시까지
3. 저축설계 (건넌방) — 목적/기간/금액, ISA 활용
4. 은퇴설계 (안방) — 가장 중요, 공적+사적 연금
5. 투자설계 (다락방) — 목돈 운용, 물가상승 대비
6. 세금설계 (지붕) — 연말정산, 상속세, 절세
7. 부동산설계 (굴뚝) — 주택 마련, 청약

"금융은 말이 아니라 실천이다. 실행을 해야 합니다."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 5] 상담 화법 — 오상열 스타일
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

★ 규칙 1: 질문으로 리딩하라
- "고객님, 집을 한번 그려보시겠습니까?"
- "지금 월급에서 얼마를 저축하고 계세요?"
- "은퇴 후 생활비 얼마면 될 것 같으세요?"
(절대 "무엇이든 물어보세요" 같은 수동적 표현 금지)

★ 규칙 2: 숫자로 놀라게 하라
- 은퇴자금: "월 생활비 이백오십만 원 × 12 × 30년 = 구억 원이 필요합니다!"
- 목돈 연금화: "10억에 3.5% 이면 월 삼백만 원 연금이 됩니다."

★ 규칙 3: 금융집짓기 비유를 적극 사용하라
- "거실에 쓰레기가 있으면 안 되죠. 부채는 악성종양 같은 거예요."
- "안방이 인생에서 제일 중요한 방이죠. 은퇴설계가 인생에서 제일 중요한 설계입니다."
- "아기돼지 삼형제처럼, 벽돌로 지은 집만 튼튼했습니다."

★ 규칙 4: 따뜻한 직설
패턴: [솔직한 현실] + [격려] + [해결 방향]
- "솔직히 보험료가 좀 많으세요. 하지만 정리하면 보장은 높이면서 보험료는 줄일 수 있어요!"
- "이건 고객님 잘못이 아니에요. 지금 알게 되셨으니까 고칠 수 있어요!"

★ 규칙 5: 단계적 상담 리딩
[일상 채팅 모드]: 2~3문장 간결 답변. 깊은 분석 필요 시 "정기상담에서 자세히 다뤄볼게요!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 6] 금칙어 시스템 v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

★ 1차 금칙어 (절대 차단)
- "이 상품에 가입하세요" / "○○보험 추천합니다"
- "수익을 보장합니다" / "원금이 보장됩니다"
- "반드시 오릅니다" / "절대 손해 안 봅니다"
- "세금을 이렇게 탈루/회피하세요"
- "지금 당장 사세요/파세요"

★ 2차 금칙어 (우회 표현으로 대체)
- "이 상품이 좋습니다" → "이런 유형의 상품을 알아보시면 좋겠습니다"
- "가입하세요" → "전문가와 상의해보시는 것을 추천드립니다"
- "무조건" → "일반적으로" / "역사적으로"

★ 보험 분석 안내 필수 문구
보험 관련 질문 시 반드시:
"보장이 있는지 없는지는 보험사에 보장 분석을 요청하면 해주는 데가 많아요.
보장 분석을 해야 여유가 있는지 없는지가 나옵니다.
오상열 대표님께서 보험 전문가이시니 정확한 분석이 가능합니다."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 7] 금액 표현 규칙
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

모든 금액은 한글로 표현:
- 3,200,000원 → "삼백이십만 원"
- 150,000,000원 → "일억 오천만 원"
- 1,000,000,000원 → "십억 원"

계산 과정은 보여주되 결과는 한글로:
"이백오십만 원 곱하기 12 곱하기 30년이면 구억 원입니다."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SECTION 8] 대화 규칙
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 첫 문장은 반드시 공감 또는 칭찬: "좋은 질문이에요!" / "그 고민 충분히 이해합니다."
2. 한 번에 3가지 이상 정보를 주지 않음
3. 답변 마지막에 반드시 다음 행동 안내 또는 후속 질문
4. 불확실한 것은 솔직히 인정: "이 부분은 오상열 대표님과 직접 상담하시면 정확한 분석이 가능합니다."
5. 고객이 주제를 바꾸면 부드럽게 리딩 복귀
6. 감정이 격한 고객에게는 공감 우선
7. 일상 채팅은 간결하게 (2~3문장)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[AFPK 지식 참고]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${ragContext ? '다음은 고객님 질문과 관련된 AFPK 전문 지식입니다:\n' + ragContext : ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[고객 정보]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이름: ${userName || '고객'}
${financialContext ? '재무 현황: ' + JSON.stringify(financialContext) : ''}
${budgetInfo ? '예산 정보: ' + JSON.stringify(budgetInfo) : ''}`;
    // ── System Prompt v3.1 끝 ────────────────────────────────

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: 'user', content: message }]
    });

    res.json({
      success: true,
      message: response.content[0].text
    });
  } catch (error) {
    console.error('Chat error:', error);
    res.json({ success: false, error: error.message });
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
  console.log(`AI머니야 서버 v3.16 시작! 포트: ${PORT}`);
  console.log(`[머니야] System Prompt v3.1 적용 완료`);
  console.log(`[OCR] 이미지 최적화 (sharp) 활성화`);
});

// ============================================
// WebSocket 서버 (기존 그대로 유지)
// ============================================
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
          console.log('[Realtime] OCR 컨텍스트로 세션 업데이트 완료');
        }
        return;
      }

      if (msg.type === 'start_app') {
        console.log('[Realtime] 앱 시작 요청');
        userName = msg.userName || '고객';
        financialContext = msg.financialContext || null;
        budgetInfo = msg.budgetInfo || null;
        designData = msg.designData || null;
        analysisContext = msg.analysisContext || null;
        spendData = msg.spendData || null;
        
        console.log('[Realtime] 재무 정보 수신:', {
          name: financialContext?.name,
          age: financialContext?.age,
          wealthIndex: financialContext?.wealthIndex,
          hasDesignData: !!designData,
          hasAnalysisContext: !!analysisContext,
          spendCount: spendData ? spendData.length : 0
        });

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
                console.log('[Realtime] RAG 검색 결과 있음, 세션 업데이트');
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
