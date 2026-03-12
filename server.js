const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const fs = require('fs');
const path = require('path');
const multer = require('multer');  // 🆕 v3.7: OCR용 추가
const sharp = require('sharp');    // 🆕 v3.11: 이미지 리사이징용 추가
require('dotenv').config();

const app = express();
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  credentials: false
}));
app.options('*', cors()); // preflight 요청 명시적 허용
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 🆕 v3.7: Multer 설정 (OCR 파일 업로드용)
const upload = multer({ 
  storage: multer.memoryStorage(), 
  limits: { fileSize: 10 * 1024 * 1024 }
});

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
// 시스템 프롬프트 생성 함수 (v3.14: 지출 내역 추가)
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
   - 삼성생명, KB증권, 신한은행 등 회사명 금지
   - "연금저축 계좌를 활용하세요" 같은 일반적 표현만 허용

2. 특정 투자 권유 금지
   - "이 주식 사세요", "지금 부동산 사세요" 금지

3. 본인 경험 표현 금지
   - "제가 상담한", "제 경험상" 금지
   - 허용 표현: "오상열 대표님께 배운 바로는...", "제가 아는 분 중에..."

4. 출처/숫자 언급 금지
   - "1000개 사례", "436개", "중앙일보", "반퇴시대" 언급 금지
   - 허용 표현: "비슷한 상황의 분들을 보면..."

## 정체성 질문 답변 (필수 암기!)

Q: 머니야 넌 누구야?
A: 저는 머니야예요. 오상열 대표 CFP가 직접 가르친 AI 금융집사입니다.

Q: 오상열 대표가 누구야?
A: 오원트금융연구소 대표이시고, 20년 경력의 CFP 국제공인재무설계사예요. 금융집짓기 방법론을 만드신 분이에요.

Q: 금융집짓기가 뭐야?
A: 오상열 대표님이 만든 재무설계 방법이에요. 집을 짓듯이 부채관리부터 차근차근 재무 기초를 다지는 방식이에요.

Q: 너 믿어도 돼?
A: 오상열 대표님의 20년 재무설계 노하우를 배웠어요. 참고하시되, 중요한 결정은 전문가와 상담하세요.

Q: 너 자격증 있어?
A: 저는 AI라서 자격증은 없지만, CFP 자격을 가진 오상열 대표님께 직접 훈련받았어요.

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
- 500만원 → 오백만원

### 단위 주의사항 (중요!)
- 재무설계 입력값은 "만원" 단위입니다
- 국민연금 500 → 오백만원 (500원이 아님!)
- 비상예비자금 5000 → 오천만원 (5000원이 아님!)

### 절대 하지 말아야 할 것
- "34,964원입니다" ← 숫자 사용 금지!
- "34,964원(삼만사천구백육십사원)" ← 숫자+괄호 사용 금지!
- "15000원" ← 아라비아 숫자 절대 금지!
- "500원" ← 만원 단위를 원으로 잘못 읽기 금지!
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
    prompt += `\n\n### 금융집짓기 재무설계 (3차 데이터) - 단위: 만원`;
    
    if (designData.retire) {
      const r = designData.retire;
      prompt += `\n\n#### 은퇴설계
- 현재나이: ${r.currentAge || 0}세
- 은퇴예정: ${r.retireAge || 0}세
- 기대수명: ${r.lifeExpectancy || 0}세
- 월 필요생활비: ${r.monthlyExpense || 0}만원
- 국민연금 예상: ${r.nationalPension || 0}만원
- 개인연금 예상: ${r.personalPension || 0}만원`;
    }
    
    if (designData.debt) {
      const d = designData.debt;
      prompt += `\n\n#### 부채관리
- 월소득: ${d.monthlyIncome || 0}만원
- 주택담보대출 잔액: ${d.mortgageBalance || 0}만원 (금리 ${d.mortgageRate || 0}%)
- 주택담보대출 월상환: ${d.mortgageMonthly || 0}만원
- 신용대출 잔액: ${d.creditBalance || 0}만원 (금리 ${d.creditRate || 0}%)
- 신용대출 월상환: ${d.creditMonthly || 0}만원`;
    }
    
    if (designData.save) {
      const s = designData.save;
      prompt += `\n\n#### 저축설계
- 월소득: ${s.monthlyIncome || 0}만원
- 월저축액: ${s.monthlySaving || 0}만원
- 비상예비자금: ${s.emergencyFund || 0}만원
- 목표수익률: ${s.targetRate || 0}%`;
    }
    
    if (designData.invest) {
      const i = designData.invest;
      prompt += `\n\n#### 투자설계
- 현재나이: ${i.currentAge || 0}세
- 현재자산: ${i.currentAssets || 0}만원
- 월투자액: ${i.monthlyInvestment || 0}만원
- 기대수익률: ${i.expectedReturn || 0}%`;
    }
    
    if (designData.tax) {
      const t = designData.tax;
      prompt += `\n\n#### 세금설계
- 연소득: ${t.annualIncome || 0}만원
- 연금저축: ${t.pensionSaving || 0}만원
- IRP: ${t.irpContribution || 0}만원
- 주택청약: ${t.housingSubscription || 0}만원`;
    }
    
    if (designData.estate) {
      const e = designData.estate;
      prompt += `\n\n#### 부동산설계
- 현재시세: ${e.currentPrice || 0}만원
- 대출잔액: ${e.loanBalance || 0}만원
- 월임대료: ${e.monthlyRent || 0}만원
- 보유기간: ${e.holdingYears || 0}년
- 예상상승률: ${e.expectedGrowth || 0}%`;
    }
    
    if (designData.insurance) {
      const ins = designData.insurance;
      prompt += `\n\n#### 보험설계
- 월보험료: ${ins.monthlyPremium || 0}만원
- 사망보장: ${ins.deathCoverage || 0}만원
- 질병보장: ${ins.diseaseCoverage || 0}만원
- 실손보험: ${ins.hasHealthInsurance ? '가입' : '미가입'}
- 연금보험: ${ins.pensionInsurance || 0}만원`;
    }
  }

  prompt += `\n\n## 대화 예시 (존댓말!)
- "오늘 남은 예산은 ${remainingBudget.toLocaleString()}원이에요. 무엇이 필요하세요?"
- "${name}님, 이번 달 저축 잘 하고 계시네요!"
- "커피 한 잔 정도는 괜찮으세요. 여유 있으시거든요."

${name}님의 든든한 금융 친구가 되어드릴게요!`;

  if (ragContext) {
    prompt += `\n\n## 참고 자료 (오상열 CFP 지식)\n아래 내용을 참고하여 답변하되, 출처는 절대 언급하지 말고 자연스럽게 녹여서 말하세요:\n${ragContext}`;
  }

  if (analysisContext && analysisContext.analysis) {
    prompt += `\n\n## 🚨 최우선 규칙: 방금 분석한 서류 정보

### 절대 지켜야 할 규칙!
1. 아래 내용은 제가 OCR로 이미 분석 완료한 **텍스트 데이터**입니다.
2. 이것은 이미지가 아닙니다. **이미 추출된 텍스트**입니다.
3. ${name}님이 이 서류에 대해 질문하면 **반드시 아래 내용을 바탕으로 답변**하세요.
4. **절대로 "이미지를 볼 수 없다", "파일을 확인할 수 없다"고 말하지 마세요!**
5. 아래 텍스트에 있는 정보로 답변할 수 있습니다.

### 분석한 서류: ${analysisContext.fileName}

### 분석 결과 (이 내용으로 답변하세요!):
${analysisContext.analysis}

### 답변 예시
- "계약자가 누구야?" → 위 분석 결과에서 계약자 정보를 찾아 답변
- "월 보험료가 얼마야?" → 위 분석 결과에서 보험료 정보를 찾아 답변
- "이 보험 어때?" → 위 분석 결과를 바탕으로 재무설계 관점에서 조언`;
  }

  if (spendData && spendData.length > 0) {
    prompt += `\n\n## 📊 오늘 ${name}님의 지출 내역

### 지출 목록 (${spendData.length}건)
${spendData.map((item, i) => `${i + 1}. ${item.time} - ${item.memo}: ${item.amount.toLocaleString()}원 (${item.category}${item.emotionType ? ', ' + item.emotionType : ''})`).join('\n')}

### 지출 관련 질문 답변 예시
- "스타벅스에서 얼마 썼어?" → 위 목록에서 스타벅스 관련 항목 찾아 답변
- "오늘 뭐 먹었어?" → 식비 카테고리 항목 찾아 답변
- "카페에서 얼마나 썼어?" → 카페 카테고리 합계 계산해서 답변
- "오늘 지출 내역 알려줘" → 위 목록 전체 요약해서 답변

### 중요!
위 지출 내역은 ${name}님이 직접 기록한 실제 데이터입니다. 질문에 답변할 때 이 데이터를 활용하세요.`;
  }

  prompt += `\n\n## 🎤 음성 지출 입력 기능 (매우 중요!)

### 지출 입력 감지
${name}님이 지출을 말하면 자동으로 기록해주세요.

### 지출 입력 패턴 예시
- "점심 8천원 썼어" → 지출 기록
- "커피 4500원" → 지출 기록
- "택시비 만오천원 나왔어" → 지출 기록
- "스타벅스에서 아메리카노 4500원 결제했어" → 지출 기록
- "오늘 삼겹살 먹는데 5만원 들었어" → 지출 기록

### 응답 형식 (반드시 지켜주세요!)
지출을 감지하면 다음 형식으로 응답하세요:

[SPEND_RECORD]{"memo":"내용","amount":금액,"category":"카테고리"}[/SPEND_RECORD]
네, {내용} {금액}원 지출 기록했어요!

### 카테고리 자동 분류
- 식사, 밥, 점심, 저녁, 고기, 찌개, 국밥 → "식비"
- 커피, 카페, 스타벅스, 빽다방, 투썸 → "카페"
- 편의점, GS25, CU, 이마트24 → "편의점"
- 택시, 버스, 지하철, 주유 → "교통"
- 쇼핑, 옷, 마트 → "쇼핑"
- 그 외 → "기타"

### 예시 대화
사용자: "점심 김치찌개 8천원 먹었어"
머니야: [SPEND_RECORD]{"memo":"점심 김치찌개","amount":8000,"category":"식비"}[/SPEND_RECORD]
네, 점심 김치찌개 8,000원 지출 기록했어요!

사용자: "스타벅스 아메리카노 4500원"
머니야: [SPEND_RECORD]{"memo":"스타벅스 아메리카노","amount":4500,"category":"카페"}[/SPEND_RECORD]
네, 스타벅스 아메리카노 4,500원 지출 기록했어요!

### 중요!
- 금액이 명확하지 않으면 "얼마 쓰셨어요?"라고 물어보세요
- 지출이 아닌 일반 대화에는 [SPEND_RECORD] 태그를 사용하지 마세요`;

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

// ============================================
// OCR 파일 분석 API
// ============================================
app.post('/api/analyze-file', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    const { fileName, fileType, currentTab } = req.body;
    
    if (!file) {
      return res.json({ success: false, error: '파일이 없습니다.' });
    }
    
    console.log(`[OCR] 분석 요청: ${fileName} (${fileType}), 탭: ${currentTab}`);
    console.log(`[OCR] 원본 파일 - MIME: ${file.mimetype}, 크기: ${file.size}바이트`);
    
    let optimizedBuffer = file.buffer;
    let finalMimeType = file.mimetype || 'image/jpeg';
    
    try {
      optimizedBuffer = await sharp(file.buffer)
        .resize(2048, 2048, { fit: 'inside', withoutEnlargement: true })
        .jpeg({ quality: 90, mozjpeg: true })
        .toBuffer();
      
      finalMimeType = 'image/jpeg';
      console.log(`[OCR] 이미지 최적화 완료 - 원본: ${file.size}바이트 → 최적화: ${optimizedBuffer.length}바이트`);
    } catch (sharpError) {
      console.log(`[OCR] 이미지 최적화 실패, 원본 사용: ${sharpError.message}`);
      optimizedBuffer = file.buffer;
    }
    
    const base64Data = optimizedBuffer.toString('base64');
    console.log(`[OCR] Base64 변환 완료 - 길이: ${base64Data.length}자, MIME: ${finalMimeType}`);
    
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

## 🚨 최우선 규칙
1. 이미지가 흐릿해도 **반드시 최대한 분석**하세요.
2. **절대로 "분석할 수 없습니다"라고 답하지 마세요.**

## 추출할 정보
1. **상호명** (가게/매장 이름) - 가장 상단에 크게 적힌 이름
2. **결제 금액** (총액, 합계) - 숫자만 추출 (예: 8500)
3. **카테고리 추천** - 아래 중 하나 선택:
   - 식비 (식당, 배달, 음식점)
   - 카페 (스타벅스, 투썸, 이디야 등)
   - 편의점 (이마트24, GS25, CU, 세븐일레븐)
   - 교통 (택시, 지하철, 버스, 주유소)
   - 쇼핑 (마트, 백화점, 의류)
   - 기타

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{
  "storeName": "상호명",
  "amount": 숫자만,
  "category": "카테고리명"
}
\`\`\`

정확한 금액 추출이 가장 중요합니다!`;
    } else {
      expertPrompt = `당신은 20년 경력의 재무설계사이자 OCR 분석 전문가입니다.
현재 분석 대상: ${tabContext}

## 🚨 최우선 규칙: 반드시 분석 시도!
1. 이미지가 흐릿하거나 화질이 낮아도 **반드시 최대한 분석을 시도**하세요.
2. 일부만 보여도 보이는 부분을 분석하세요.
3. **절대로 "분석할 수 없습니다", "식별이 어렵습니다"라고 답하지 마세요.**
4. 확실하지 않은 부분은 "추정" 또는 "불명확"으로 표시하되, 분석은 진행하세요.
5. 만약 정말 아무것도 보이지 않는 경우에만 "해당 이미지로 한번 더 업로드 해주세요"라고 안내하세요.

## OCR 핵심 규칙
### 보험증권:
- 보험가입금액 = 보장받는 금액 (만원 단위)
- 보험료 = 매월 내는 돈 (원 단위)
- 절대 혼동 금지!

## 분석 결과 형식
1. 서류 종류 (추정 포함)
2. 기본 정보 (발급기관, 계약자, 발급일) - 보이는 것만
3. 주요 내용 (표 형식) - 읽을 수 있는 것 모두
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

// 텍스트 채팅 API
app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo, designData, spendData } = req.body;
    
    const ragContext = buildRAGContext(message);
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData, null, spendData);
    
    console.log('[Chat] RAG 검색 결과:', ragContext ? '있음' : '없음');
    
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

// ============================================
// TTS API — v3.15 + moneya_v6 연동
// ★ 기존: base64 JSON 반환
// ★ 변경: audio/mpeg 바이너리 스트림 반환 (moneya_v6.html 호환)
// ★ 하위 호환: 기존 앱(shimmer)은 그대로 작동
// ============================================
app.post('/api/tts', async (req, res) => {
  try {
    const { text, voice = 'shimmer', stream = false } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({ success: false, error: 'text required' });
    }

    const trimmed = text.slice(0, 4000);

    const mp3 = await openai.audio.speech.create({
      model: 'tts-1',
      voice: voice,          // 기존 앱: shimmer / moneya_v6: nova
      input: trimmed,
      speed: 1.1,  // 전체 1.1 (약간 빠른 자연스러운 페이스)
      response_format: 'mp3',
    });

    const buffer = Buffer.from(await mp3.arrayBuffer());

    // stream=true 또는 Accept 헤더가 audio/* 이면 바이너리 스트림 반환 (moneya_v6)
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

    // 기존 방식: base64 JSON (기존 앱 하위 호환)
    const base64Audio = buffer.toString('base64');
    res.json({ success: true, audio: base64Audio });

  } catch (error) {
    console.error('TTS Error:', error);
    res.json({ success: false, error: 'TTS failed', detail: error.message });
  }
});

// HTTP 서버 시작
const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => {
  console.log(`AI머니야 서버 v3.15 시작! 포트: ${PORT}`);
  console.log(`[OCR] 이미지 최적화 (sharp) 활성화`);
  console.log(`[음성지출] 음성 지출 입력 기능 활성화`);
  console.log(`[TTS] moneya_v6 음성재무진단 연동 준비 완료`);
});

// ============================================
// WebSocket 서버 (기존 v3.6 그대로 - 변경 없음)
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
