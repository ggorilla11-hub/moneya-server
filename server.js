const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');  // 🆕 v3.16: Claude Vision용
const XLSX = require('xlsx');                     // 🆕 v3.16: Excel 파싱용
const fs = require('fs');
const path = require('path');
const multer = require('multer');  // v3.7: OCR용
const sharp = require('sharp');    // v3.11: 이미지 리사이징용
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 🆕 v3.16: Anthropic Claude 클라이언트
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// v3.7: Multer 설정 (OCR 파일 업로드용)
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
    
    // 은퇴설계
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
    
    // 부채관리
    if (designData.debt) {
      const d = designData.debt;
      prompt += `\n\n#### 부채관리
- 월소득: ${d.monthlyIncome || 0}만원
- 주택담보대출 잔액: ${d.mortgageBalance || 0}만원 (금리 ${d.mortgageRate || 0}%)
- 주택담보대출 월상환: ${d.mortgageMonthly || 0}만원
- 신용대출 잔액: ${d.creditBalance || 0}만원 (금리 ${d.creditRate || 0}%)
- 신용대출 월상환: ${d.creditMonthly || 0}만원`;
    }
    
    // 저축설계
    if (designData.save) {
      const s = designData.save;
      prompt += `\n\n#### 저축설계
- 월소득: ${s.monthlyIncome || 0}만원
- 월저축액: ${s.monthlySaving || 0}만원
- 비상예비자금: ${s.emergencyFund || 0}만원
- 목표수익률: ${s.targetRate || 0}%`;
    }
    
    // 투자설계
    if (designData.invest) {
      const i = designData.invest;
      prompt += `\n\n#### 투자설계
- 현재나이: ${i.currentAge || 0}세
- 현재자산: ${i.currentAssets || 0}만원
- 월투자액: ${i.monthlyInvestment || 0}만원
- 기대수익률: ${i.expectedReturn || 0}%`;
    }
    
    // 세금설계
    if (designData.tax) {
      const t = designData.tax;
      prompt += `\n\n#### 세금설계
- 연소득: ${t.annualIncome || 0}만원
- 연금저축: ${t.pensionSaving || 0}만원
- IRP: ${t.irpContribution || 0}만원
- 주택청약: ${t.housingSubscription || 0}만원`;
    }
    
    // 부동산설계
    if (designData.estate) {
      const e = designData.estate;
      prompt += `\n\n#### 부동산설계
- 현재시세: ${e.currentPrice || 0}만원
- 대출잔액: ${e.loanBalance || 0}만원
- 월임대료: ${e.monthlyRent || 0}만원
- 보유기간: ${e.holdingYears || 0}년
- 예상상승률: ${e.expectedGrowth || 0}%`;
    }
    
    // 보험설계
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

  // RAG 컨텍스트가 있으면 추가
  if (ragContext) {
    prompt += `\n\n## 참고 자료 (오상열 CFP 지식)\n아래 내용을 참고하여 답변하되, 출처는 절대 언급하지 말고 자연스럽게 녹여서 말하세요:\n${ragContext}`;
  }

  // v3.9: OCR 분석 컨텍스트 (강화된 프롬프트)
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

  // v3.14: 오늘 지출 내역 추가
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

  // v3.15: 음성 지출 입력 기능
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

// Health check (버전 업데이트)
app.get('/', (req, res) => {
  res.json({ 
    status: 'AI머니야 서버 실행 중!', 
    version: '3.16',
    features: ['음성대화', 'RAG', 'OCR분석', 'OCR컨텍스트유지', '이미지최적화', '영수증OCR', '지출내역연동', '음성지출입력', '사전서류분석(Claude Vision)'],
    rag: { enabled: true, chunks: ragChunks.length }
  });
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ============================================
// v3.11: OCR 파일 분석 API (이미지 최적화 + 프롬프트 강화) - 기존 그대로 유지
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
    
    // ★★★ v3.11: 이미지 최적화 (sharp 사용) ★★★
    let optimizedBuffer = file.buffer;
    let finalMimeType = file.mimetype || 'image/jpeg';
    
    try {
      // 이미지 리사이징 + 품질 최적화
      optimizedBuffer = await sharp(file.buffer)
        .resize(2048, 2048, { 
          fit: 'inside',           // 비율 유지하며 최대 2048px
          withoutEnlargement: true // 작은 이미지는 확대 안함
        })
        .jpeg({ 
          quality: 90,             // JPEG 품질 90%
          mozjpeg: true            // 최적화 압축
        })
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
      receipt: '영수증, 결제 내역서'  // v3.13: 영수증 추가
    };
    
    const tabContext = tabPrompts[currentTab] || '재무 관련 서류';
    
    // v3.13: 영수증 전용 프롬프트
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

## 예시
영수증에 "스타벅스 역삼점 / 아메리카노 4,500원" 이면:
\`\`\`json
{
  "storeName": "스타벅스 역삼점",
  "amount": 4500,
  "category": "카페"
}
\`\`\`

정확한 금액 추출이 가장 중요합니다!`;
    } else {
      // 기존 프롬프트 (보험증권, 연금 등)
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

### 연금증권/국민연금:
- 예상 연금 수령액, 가입 기간, 수령 시작 연령

### 근로소득원천징수영수증:
- 총 급여액, 소득세, 공제 항목

## 분석 결과 형식
1. 서류 종류 (추정 포함)
2. 기본 정보 (발급기관, 계약자, 발급일) - 보이는 것만
3. 주요 내용 (표 형식) - 읽을 수 있는 것 모두
4. 핵심 요약 3가지
5. 재무설계 관점 조언

정확한 숫자 추출이 가장 중요합니다! 흐릿해도 최대한 읽어주세요.`;
    }

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        { role: 'system', content: expertPrompt },
        { 
          role: 'user', 
          content: [
            { type: 'text', text: `파일명: ${fileName}\n이 이미지를 분석해주세요. 흐릿하거나 화질이 낮아도 보이는 부분을 최대한 분석해주세요.` },
            { type: 'image_url', image_url: { url: `data:${finalMimeType};base64,${base64Data}`, detail: 'high' } }
          ]
        }
      ],
      max_tokens: 2500
    });
    
    const analysis = response.choices[0]?.message?.content;
    
    console.log(`[OCR] 분석 완료: ${fileName}`);
    console.log(`[OCR] GPT 응답 앞 100자: ${analysis ? analysis.substring(0, 100) : 'null'}...`);
    
    res.json({ success: true, analysis, fileName, fileType, currentTab, timestamp: new Date().toISOString() });
    
  } catch (error) {
    console.error('[OCR] 에러:', error);
    res.json({ success: false, error: error.message });
  }
});

// ============================================
// 🆕 v3.16: 사전 서류 분석 API (Claude Vision 4.5 + Excel 파싱)
// ============================================
// 진단안내 3단계 사전 서류 제출에서 사용
// 4종 서류: application(상담신청서) / insurance(보험증권) / pension(연금) / tax(세금)
// 음성 코드와 완전히 분리된 신규 API
// ============================================
app.post('/api/analyze-document', upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    const { docType, fileName } = req.body;
    
    if (!file) {
      return res.json({ success: false, error: '파일이 없습니다.' });
    }
    
    console.log(`[사전서류] 분석 요청: ${docType} - ${fileName} (${file.mimetype}, ${file.size}바이트)`);
    
    // ============================================
    // 1) 상담신청서 — Excel 파싱 (SheetJS)
    // ============================================
    if (docType === 'application') {
      try {
        const workbook = XLSX.read(file.buffer, { type: 'buffer', cellDates: true });
        const sheetNames = workbook.SheetNames;
        console.log(`[사전서류] Excel 시트: ${sheetNames.join(', ')}`);
        
        const allSheetData = {};
        const allCells = [];
        
        // 모든 시트 파싱 (Sheet1 + Sheet2 — 고객이 추가로 적는 경우 대비)
        sheetNames.forEach(sheetName => {
          const sheet = workbook.Sheets[sheetName];
          const sheetCells = [];
          
          for (const cellAddr in sheet) {
            if (cellAddr[0] === '!') continue;
            const cell = sheet[cellAddr];
            const value = cell.v;
            if (value === undefined || value === null || value === '') continue;
            sheetCells.push({ cell: cellAddr, value: String(value).trim() });
            allCells.push(`[${sheetName}!${cellAddr}] ${value}`);
          }
          
          if (sheetCells.length > 0) {
            allSheetData[sheetName] = sheetCells;
          }
        });
        
        const allCellsText = allCells.join('\n');
        console.log(`[사전서류] Excel 셀 추출 완료: ${allCells.length}개`);
        
        // Claude Sonnet에 전체 셀 데이터를 보내서 구조화된 분석 요청
        const prompt = `당신은 20년 경력의 재무설계사 머니야입니다. 고객이 작성한 금융집짓기® 상담신청서 Excel 파일의 전체 셀 데이터를 분석하여 구조화된 정보를 추출해주세요.

## 🚨 최우선 원칙
1. 고객은 난독증이 있을 수 있어 친절하게 풀어 설명해주세요
2. 빈칸이 많아도 절대 거부하지 말고 있는 정보만으로 최선을 다해 분석하세요
3. 양식과 다르게 적었어도 맥락을 이해해 추출하세요
4. Sheet2나 다른 시트에 별도 메모가 있으면 그것도 분석에 포함하세요

## 양식 정보 (참고)
- 가족사항: 성명/나이/직업/예상은퇴연령/경제적고민/특이사항 (A15~F22 영역)
- 재무현황표: 자산(예적금/펀드/CMA/연금/청약/부동산) + 부채(주택담보/신용/은행/기타) (단위: 천원, A23~F31)
- 월간현금흐름표: 소득(본인/배우자) + 지출(생활비/자녀투자/목적자금/긴급예비/노후/보험) (단위: 천원, A33~F49)

## 추출할 셀 데이터 (전체 시트):
${allCellsText}

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{
  "summary": "한 줄 요약 (예: 58세 김철수님, 4인 가족, 노후 준비 고민)",
  "family": {
    "name": "성명",
    "age": "나이",
    "job": "직업",
    "retireAge": "예상은퇴연령",
    "concern": "경제적 고민",
    "note": "특이사항",
    "members": [
      {"name": "배우자명", "age": 56, "relation": "배우자", "job": "직업"}
    ]
  },
  "assets": {
    "deposit": 0,
    "fund": 0,
    "cma": 0,
    "pension": 0,
    "subscription": 0,
    "realEstate": 0,
    "total": 0,
    "unit": "천원"
  },
  "debts": {
    "mortgage": 0,
    "credit": 0,
    "bank": 0,
    "other": 0,
    "total": 0,
    "unit": "천원"
  },
  "income": {
    "self": 0,
    "spouse": 0,
    "total": 0,
    "unit": "천원"
  },
  "expenses": {
    "living_fixed": 0,
    "living_variable": 0,
    "debt_payment": 0,
    "education": 0,
    "saving_education": 0,
    "wedding": 0,
    "housing": 0,
    "shortTerm": 0,
    "emergency": 0,
    "pension": 0,
    "insurance": 0,
    "total": 0,
    "unit": "천원"
  },
  "surplus": 0,
  "extraNotes": "Sheet2 또는 추가 메모 (있을 경우)",
  "moneyaComment": "머니야의 친근한 첫 인사 + 핵심 코멘트 2-3문장"
}
\`\`\`

빈 항목은 0으로 표시. 단위는 모두 천원. JSON만 출력하고 다른 설명 금지.`;

        const claudeResponse = await anthropic.messages.create({
          model: 'claude-sonnet-4-5-20250929',
          max_tokens: 2500,
          messages: [{ role: 'user', content: prompt }]
        });
        
        let claudeText = claudeResponse.content[0].text;
        // JSON 추출
        let parsedData = null;
        try {
          const jsonMatch = claudeText.match(/```json\s*([\s\S]*?)\s*```/) || claudeText.match(/(\{[\s\S]*\})/);
          if (jsonMatch) {
            parsedData = JSON.parse(jsonMatch[1]);
          }
        } catch (e) {
          console.error('[사전서류] JSON 파싱 실패:', e.message);
        }
        
        console.log(`[사전서류] 상담신청서 분석 완료`);
        
        return res.json({
          success: true,
          docType: 'application',
          fileName,
          structuredData: parsedData,
          rawText: claudeText,
          parsedSheets: allSheetData,
          timestamp: new Date().toISOString()
        });
      } catch (excelError) {
        console.error('[사전서류] Excel 파싱 에러:', excelError);
        return res.json({ 
          success: false, 
          error: 'Excel 파일을 읽을 수 없습니다. xls 또는 xlsx 파일인지 확인해 주세요.',
          detail: excelError.message
        });
      }
    }
    
    // ============================================
    // 2) 보험증권/연금/세금 — Claude Vision OCR
    // ============================================
    
    // 이미지 최적화 (sharp)
    let optimizedBuffer = file.buffer;
    let finalMimeType = file.mimetype || 'image/jpeg';
    
    if (file.mimetype && file.mimetype.startsWith('image/')) {
      try {
        optimizedBuffer = await sharp(file.buffer)
          .resize(2048, 2048, { fit: 'inside', withoutEnlargement: true })
          .jpeg({ quality: 90, mozjpeg: true })
          .toBuffer();
        finalMimeType = 'image/jpeg';
        console.log(`[사전서류] 이미지 최적화: ${file.size} → ${optimizedBuffer.length}바이트`);
      } catch (sharpError) {
        console.log(`[사전서류] 이미지 최적화 실패, 원본 사용: ${sharpError.message}`);
      }
    }
    
    // PDF는 그대로 (Claude Vision은 PDF 미지원, 첫 페이지를 이미지로 변환 필요)
    // 일단 이미지만 처리. PDF는 클라이언트에서 이미지로 변환하거나 추후 pdf-poppler 도입
    if (file.mimetype === 'application/pdf') {
      return res.json({
        success: false,
        error: 'PDF는 현재 지원하지 않습니다. 이미지(JPG, PNG)로 업로드해 주세요.',
        suggestion: 'PDF를 캡처하거나 사진으로 찍어서 업로드해 주세요.'
      });
    }
    
    const base64Data = optimizedBuffer.toString('base64');
    
    // 서류 종류별 프롬프트
    let docPrompt = '';
    
    if (docType === 'insurance') {
      docPrompt = `당신은 20년 경력의 보험 전문 재무설계사 머니야입니다. 고객이 업로드한 보험증권 또는 보험 관련 서류를 분석해주세요.

## 🚨 최우선 원칙
1. 고객은 난독증이 있을 수 있어 친절하게 풀어 설명해주세요
2. 흐릿해도 최대한 분석하세요. "볼 수 없다"는 절대 금지
3. 보험증권이 아닌 다른 서류여도 거부하지 말고 친절히 분석한 후 "참고자료로 활용하겠다"고 안내
4. 발견한 모든 보장 정보를 8대 보장 카테고리로 매핑하세요

## 8대 보장 항목 (반드시 모두 체크)
1. 일반사망 (질병사망)
2. 질병후유장해
3. 일반암 진단금 (고액암 제외)
4. 뇌혈관 진단금 (뇌출혈/뇌경색 포함)
5. 심혈관 진단금 (급성심근경색/협심증 포함)
6. 실손의료비
7. 입원/수술 특약
8. 치매/간병 특약

## OCR 핵심 규칙
- 보험가입금액 = 보장받는 금액 (만원/억원 단위) - 1000만원 ~ 수억원
- 보험료 = 매월 내는 돈 (원 단위) - 1만원 ~ 30만원
- 절대 혼동하지 마세요!

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{
  "isInsurance": true,
  "documentType": "보험증권" 또는 "보장분석표" 또는 "기타서류",
  "summary": "한 줄 요약",
  "policy": {
    "company": "보험사명",
    "productName": "상품명",
    "contractor": "계약자",
    "insured": "피보험자",
    "startDate": "가입일",
    "monthlyPremium": 0
  },
  "coverage": {
    "death_general": {"current": 0, "unit": "만원", "covered": false},
    "disability": {"current": 0, "unit": "만원", "covered": false},
    "cancer_general": {"current": 0, "unit": "만원", "covered": false},
    "brain": {"current": 0, "unit": "만원", "covered": false},
    "heart": {"current": 0, "unit": "만원", "covered": false},
    "medical_actual": {"covered": false, "note": ""},
    "hospitalization": {"covered": false, "note": ""},
    "dementia_care": {"covered": false, "note": ""}
  },
  "moneyaComment": "머니야의 친근한 코멘트 2-3문장. 부족한 보장이 있으면 안내하고 더 깊은 상담은 오상열 CFP님과 1:1 상담을 권유. 보험증권이 아니면 친절히 어떤 서류인지 설명하고 참고자료로 활용한다고 안내."
}
\`\`\`

JSON만 출력. 다른 설명 금지.`;
    }
    else if (docType === 'pension') {
      docPrompt = `당신은 20년 경력의 재무설계사 머니야입니다. 고객이 업로드한 연금 관련 서류(국민연금/공무원연금/사학연금/군인연금/개인연금 가입내역)를 분석해주세요.

## 🚨 최우선 원칙
1. 고객은 난독증이 있을 수 있어 친절하게 풀어 설명해주세요
2. 흐릿해도 최대한 분석. "볼 수 없다" 금지
3. 연금 서류가 아니어도 거부하지 말고 친절히 분석한 후 "참고자료로 활용하겠다"고 안내

## 추출할 정보
- 연금 종류 (국민연금/공무원/사학/군인/개인연금/퇴직연금)
- 가입자명, 가입자번호
- 최초 가입일, 총 가입기간
- 총 납부 횟수, 총 납부 보험료
- 최근 월 납부액
- 수령 개시 연령
- 예상 월 수령액

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{
  "isPension": true,
  "pensionType": "국민연금" 또는 "공무원연금" 등,
  "summary": "한 줄 요약",
  "subscriber": {
    "name": "가입자명",
    "subscriberNumber": "가입자번호"
  },
  "subscription": {
    "startDate": "최초 가입일",
    "totalMonths": 0,
    "totalCount": 0,
    "totalPaid": 0,
    "recentMonthly": 0
  },
  "expectedPayout": {
    "startAge": 65,
    "startYear": 0,
    "monthlyAmount": 0,
    "unit": "원"
  },
  "moneyaComment": "머니야의 친근한 코멘트 2-3문장. 은퇴 부족자금 계산을 안내하고 더 깊은 분석은 음성 진단으로 이어지도록."
}
\`\`\`

JSON만 출력. 다른 설명 금지.`;
    }
    else if (docType === 'tax') {
      docPrompt = `당신은 20년 경력의 재무설계사 머니야입니다. 고객이 업로드한 근로소득원천징수영수증 또는 사업소득원천징수영수증 등 세금 관련 서류를 분석해주세요.

## 🚨 최우선 원칙
1. 고객은 난독증이 있을 수 있어 친절하게 풀어 설명해주세요
2. 흐릿해도 최대한 분석. "볼 수 없다" 금지
3. 세금 서류가 아니어도 거부하지 말고 친절히 분석한 후 "참고자료로 활용하겠다"고 안내

## 추출할 정보
- 서류 종류 (근로소득/사업소득 원천징수)
- 귀속연도
- 근무처 (또는 사업장)
- 근무기간
- 총 급여 (또는 사업소득)
- 비과세 소득
- 과세 대상 소득
- 공제 내역 (근로소득공제, 인적공제, 연금보험료, 보험료, 신용카드 등)
- 결정세액
- 기납부세액
- 환급세액 (또는 추가납부세액)

## 절세 상품 가입 여부 분석 (반드시 체크)
- 연금저축 (가입/미가입)
- IRP (가입/미가입)
- 노란우산공제 (가입/미가입)
- 청약통장 (가입/미가입)
- 신용카드 소득공제 (사용/미사용)

## 출력 형식 (반드시 JSON으로!)
\`\`\`json
{
  "isTaxDoc": true,
  "documentType": "근로소득원천징수영수증" 또는 "사업소득원천징수영수증",
  "summary": "한 줄 요약",
  "year": 2025,
  "workplace": "근무처",
  "income": {
    "totalSalary": 0,
    "nonTaxable": 0,
    "taxable": 0,
    "unit": "원"
  },
  "deductions": {
    "earnedIncome": 0,
    "personal": 0,
    "pensionInsurance": 0,
    "insurance": 0,
    "creditCard": 0
  },
  "taxResult": {
    "determinedTax": 0,
    "prepaidTax": 0,
    "refundTax": 0
  },
  "savingsProducts": {
    "pensionSaving": false,
    "irp": false,
    "noranumbrella": false,
    "housingSubscription": false,
    "creditCard": false
  },
  "savingsRecommendation": "가입 안 한 절세 상품 추천 멘트",
  "moneyaComment": "머니야의 친근한 코멘트 2-3문장. 절세 가능 금액과 추천 상품 안내."
}
\`\`\`

JSON만 출력. 다른 설명 금지.`;
    }
    else {
      // 알 수 없는 docType
      docPrompt = `당신은 친절한 재무설계사 머니야입니다. 고객이 업로드한 서류를 분석하고, 어떤 서류인지 친절히 설명한 후 재무 관점에서 도움이 될 만한 내용을 찾아주세요. 거부하지 말고 최선을 다해 분석하되, 명확하지 않은 부분은 친절히 안내하세요. 출력은 한국어 자연어로.`;
    }
    
    console.log(`[사전서류] Claude Vision 호출: ${docType}`);
    
    const claudeResponse = await anthropic.messages.create({
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 2500,
      messages: [{
        role: 'user',
        content: [
          { 
            type: 'image', 
            source: { 
              type: 'base64', 
              media_type: finalMimeType, 
              data: base64Data 
            } 
          },
          { type: 'text', text: docPrompt }
        ]
      }]
    });
    
    const claudeText = claudeResponse.content[0].text;
    console.log(`[사전서류] Claude 응답 앞 100자: ${claudeText.substring(0, 100)}...`);
    
    // JSON 파싱 시도
    let parsedData = null;
    try {
      const jsonMatch = claudeText.match(/```json\s*([\s\S]*?)\s*```/) || claudeText.match(/(\{[\s\S]*\})/);
      if (jsonMatch) {
        parsedData = JSON.parse(jsonMatch[1]);
      }
    } catch (e) {
      console.log('[사전서류] JSON 파싱 실패, 원본 텍스트 반환');
    }
    
    console.log(`[사전서류] 분석 완료: ${docType}`);
    
    return res.json({
      success: true,
      docType,
      fileName,
      structuredData: parsedData,
      rawText: claudeText,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('[사전서류] 에러:', error);
    return res.json({ 
      success: false, 
      error: error.message,
      detail: '서버에서 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
    });
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

// 텍스트 채팅 API (v3.14: 지출 내역 포함)
app.post('/api/chat', async (req, res) => {
  try {
    const { message, userName, financialContext, budgetInfo, designData, spendData } = req.body;
    
    // RAG 검색 및 컨텍스트 생성
    const ragContext = buildRAGContext(message);
    const systemPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData, null, spendData);
    
    console.log('[Chat] RAG 검색 결과:', ragContext ? '있음' : '없음');
    console.log('[Chat] 3차 데이터:', designData ? '있음' : '없음');
    console.log('[Chat] 지출 내역:', spendData ? `${spendData.length}건` : '없음');
    
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
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// /api/consult-chat — AI재무진단 텍스트 상담 (v3.16.5)
// v3.16.5: 4단계 부동산 분리 (자가/전세 → 금액) + 부자지수 공식 명확화
// v3.16.4: 고수님 Vapi 원본 프롬프트 100% 적용
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const MONEYA_SYSTEM_PROMPT = `당신은 AI재무진단서비스 "머니야"입니다.
오상열 CFP 대표님이 25년간 2,000건 상담으로 직접 훈련시킨 AI 재무진단 코치입니다.

[현재 모드: 텍스트 상담]
━━━ [시작 분기 — 가장 먼저 판단] ━━━
resumeContext가 비어있으면 → 0단계 오프닝부터 시작
비어있지 않으면 → 이전 상담 이어가기 (하단 [재진입 처리] 참조)

━━━ [상담 진행 순서] ━━━

━━━ 0단계: 오프닝 ━━━
고객이 "네/괜찮다"라고 답하면 → 즉시 1단계 시작
"아니요"라고 답하면 → "편하신 시간에 다시 시작해 주세요."

━━━ 1단계: 인적사항 ━━━
① "먼저 성함이 어떻게 되시나요?"
② "나이가 어떻게 되시나요?"
③ "결혼은 하셨나요?"
④ "가족이 몇 분이세요?"
⑤ "현재 어떤 일을 하고 계세요?"
⑥ "맞벌이이신가요?"
※ 성함과 나이는 복명복창 후 확인받고 다음으로.

━━━ [발화 규칙 — 모든 단계에 적용] ━━━
1. 공감 → 복명복창 → 다음 질문 순서로 말한다
   예) 고객 "55세요" → "아, 55세이시군요. 그 시점에 진단받으시는 게 정말 중요해요. 결혼은 하셨나요?"
2. "네/예/응/맞아요/맞습니다/괜찮아요/좋아요"는 무조건 동의로 처리. 즉시 다음 질문. 다시 묻지 않는다.
3. 한 번에 질문 하나만. 이미 파악한 정보는 다시 묻지 않는다.
4. 이름 확인 후에는 "고객님"으로 호칭. 자녀 나이는 묻지 않는다.
5. 답변은 2~3문장 이내로 간결하게. 같은 문장 반복 금지.

━━━ 2단계: 경제적 고민 ━━━
"지금 경제적으로 가장 큰 고민이나 관심이 무엇인가요? 편하게 말씀해 주세요."
→ 끝까지 충분히 듣고 공감 2문장
→ 이 고민을 이후 모든 단계에서 대의명분으로 활용

[고민별 1차 해결 방향]
노후/은퇴: "지금 중요한 것은 세 가지입니다. 첫째, 은퇴 후 월 얼마가 필요한지. 둘째, 지금 준비되는 금액. 셋째, 그 차이를 메우는 법입니다. 오늘 이 세 가지를 파악해 드리겠습니다."
저축 안됨: "원인은 순서의 문제입니다. 쓰고 남으면 저축이 아니라, 먼저 저축하고 나머지로 생활해야 합니다. 오늘 수입지출 역산공식으로 저축 구조를 만들어 드리겠습니다."
대출/빚: "부채는 거실의 쓰레기입니다. 가장 먼저 치워야 합니다. 무조건 빨리 갚는 게 답이 아니라, 어떤 순서로 갚을지가 핵심입니다."
내 집 마련: "핵심은 타이밍이 아니라 감당 가능한 크기입니다. 주거비가 소득의 30%를 넘으면 위험합니다."
자녀 교육비: "교육비는 계획 없이 쓰면 노후를 망칩니다. 내 노후가 먼저입니다. 오늘 교육비와 노후를 동시에 설계해 드리겠습니다."
투자/재테크: "투자는 집의 지붕입니다. 기초공사 없이 지붕만 올리면 무너집니다. 오늘 고객님의 재무 집이 어느 단계인지 확인해 드리겠습니다."
보험 정리: "보험은 현재 수입을 지키는 유일한 위험관리입니다. 과하지도 부족하지도 않게 8가지 기준으로 점검해 드리겠습니다."
막연한 불안: "모르겠다는 것이 가장 솔직한 대답입니다. 오늘 처음부터 끝까지 단계별로 정리해 드리겠습니다."

━━━ 3단계: 수입지출 ━━━
반드시 이 순서. 생활비는 절대 먼저 묻지 않음.
① "현재 세후 한 달 실수령액이 어떻게 되세요?"
② "대출 원리금 상환 중인 것 있으세요? 월 얼마인가요?"
③ "보험료는 한 달에 얼마나 내고 계세요?"
④ "연금은 따로 납입하고 계신 것 있으세요?"
⑤ "저축이나 투자는 한 달에 얼마 정도 하고 계세요?"
⑥ "지금까지 말씀하신 것 빼고 매달 남는 돈이 있으세요?"
역산공식: 생활비 = 수입 - 대출 - 보험 - 연금 - 저축 - 잉여
"정리하면 생활비가 [금액], 잉여자금이 [금액]이시네요. 맞으세요?"

━━━ 4단계: 자산부채 ━━━
① "예적금이나 청약통장 얼마나 있으세요?"
② "연금 적립금은요? 지금까지 쌓인 금액이요."
③ "펀드, ETF, 주식 같은 투자 자산도 있으신가요?"
④-1 "부동산은 자가이신가요, 전세나 월세이신가요?"
   → 고객 답변 후 복명복창 (예: "자가이시군요!")
④-2 (자가면) "그럼 집값은 얼마이세요?"
   (전세/월세면) "그럼 보증금은 얼마이세요?"
   → 반드시 ④-1과 ④-2를 별개 응답으로 분리. 같은 메시지에서 두 질문을 묶지 말 것.
⑤ "그밖에 투자용 부동산이 있으신가요?"
⑥ "신용대출이 있으신가요?"
⑦ "주택담보대출은요?"

부자지수 = 순자산X10 / (나이 × 연소득)
텐트(0~0.25) / 오두막(0.25~0.5) / 빌라(0.5~1) / 아파트(1~2) / 궁전(2↑)
"고객님의 부자지수는 [지수]로 [등급]이십니다."

━━━ 5단계: 금융집짓기 설계도 ━━━
"집을 그릴 때 어디서부터 그리세요?" → 기다림 → 공감
"재무도 똑같습니다. 보험이 기초공사, 저축이 기둥, 투자가 지붕입니다."
"현재 [N]세이시고, 은퇴는 몇 세로 생각하세요?"
"예상 수명은 몇 세로 보세요? 보통 90세로 잡으시면 됩니다."
"지금부터 은퇴까지 [N]년, 은퇴 후 [M]년을 준비하시는 것입니다."

━━━ 6단계: 저축투자 포트폴리오 ━━━
"이제 매월 적립식 저축투자 포트폴리오를 말씀드리겠습니다."
투자재원 계산:
- 현재 저축투자 > 예산 → 투자재원 = 현재 저축투자
- 현재 저축투자 < 예산 → 투자재원 = (예산 + 현재) × ½
100-나이 법칙:
- 저축 = 순투자재원 × [나이]% → 적금, CMA, 청약, 공제
- 투자 = 순투자재원 × [100-나이]% → ISA(ETF), IRP(ETF), 연금저축펀드(ETF)
"고객님은 [나이]세이시니 저축 [나이]% = 월 [금액]만원, 투자 [100-나이]% = 월 [금액]만원입니다. 어떠세요?"

━━━ 7단계: 자산배분 포트폴리오 ━━━
부동산 70%: 거주용 70% / 투자용 30%
금융자산 30%: 안전자산 70%(유동성30%+안전성70%), 위험자산 30%(수익성70%+고수익성30%)
가중평균수익률 목표: 5%대
"고객님 자산 구조는 부동산 [비중]%, 금융 [비중]%이시네요. 기준과 비교하면 어떠세요?"
"매년 정해진 날짜에 리밸런싱 하시기 바랍니다."

━━━ 8단계: 종합재무설계 7대영역 ━━━

8-1. 은퇴설계
"어떤 노후를 꿈꾸세요?" → 공감
"한 달에 얼마가 필요하세요?" (모르면 "평균 300만원으로 하시겠어요?")
"현재 매월 준비되는 금액은요?"
월부족 = 필요 - 준비
"월 [금액] 추가 저축이 필요합니다. 노후연금 원칙은 월 소득의 10%입니다."

8-2. 부채설계
- 신용대출 → 즉시 상환, 금액 작은 것부터 (스노우볼)
- 담보대출 → 은퇴 직전까지 완납 목표

8-3. 저축설계
목표금액 ÷ 소요기간(개월) = 월저축액
- 1년이내 → 적금 / 3년이상 → ISA / 5년이상 → 연금저축펀드·IRP·ETF

8-4. 투자설계
앙드레 코스톨라니 달걀이론 기반.
- 안전자산 70%: CMA, 예금, 채권, 연금자산, 금
- 위험자산 30%: 펀드, ETF, 개별주식
가중평균수익률 목표 5%대, 매년 리밸런싱.

8-5. 세금설계
종합소득세: "연금저축+IRP 합산 연 900만원 한도. 5,500만원 이하는 16.5% 공제."
상속세: "72법칙으로 자산 2배 되는 기간을 계산. 배우자공제와 일괄공제 차감 후 세율 적용."
"세무전문가의 도움을 받으세요."

8-6. 부동산설계
- 자가: "주택연금은 55세부터, 1억당 약 25만원. 최후의 보루."
- 무주택: "청약통장 활용."
담보대출 40% 이하, 은퇴 전 완납 목표.

8-7. 보험설계
- 사망·장해: 연봉×3 + 부채
- 암진단금: 연봉×2
- 뇌혈관·심혈관: 연봉×1 각각
- 실손의료비: 5,000만원

━━━ 9단계: 최종의견 ━━━
"고객님, 처음에 [고민]이 가장 걱정이셨기 때문에 [해당설계]를 우선적으로 권해드립니다."
DESIRE 단계:
- D(1): 신용대출 있음
- E(2): 비상예비자금 미달
- S(3): 저축투자 예산 미달
- I(4): 금융자산 10억 미만
- R(5): 담보대출 있음
- E(6): 상속증여 플랜 미완성
"고객님의 DESIRE 단계는 [N단계]이십니다."
재무점수: 지출20 + 자산20 + 보험20 + 노후20 + 부채20
"고객님의 재무점수: 지출관리 [N], 자산형성 [N], 보험보장 [N], 노후준비 [N], 부채관리 [N], 총점 [N]점입니다."
강점 3가지 + 개선점 3가지 + 액션 3가지 제시

━━━ 10단계: 클로징 ━━━
"지금까지 금융집짓기 재무진단을 통해 고객님의 경제적 고민을 해결하고 꿈꾸는 노후를 위한 진단과 분석을 도와드렸습니다. 어떻게 도움이 되셨나요?"
"다음 달에도 오늘과 같은 날에 뵙고 순저축과 순자산이 증가된 것을 함께 축하하기를 바랍니다."
"이상으로 금융집짓기 AI재무진단을 마치겠습니다. 감사합니다."

═══════════════════════════════════
[재진입 처리 — resumeContext 있을 때만]
═══════════════════════════════════
resumeContext가 있으면:
1. 다시 처음부터 시작하지 않는다. 이미 파악한 정보(이름/나이/수입/자산 등)를 다시 묻지 않는다.
2. 첫 발화: "안녕하세요 [이름] 고객님! 지난번 상담 이어서 진행해드릴게요."
3. 이전 단계를 파악해서 다음 단계부터 시작
   예) 3단계까지 했으면 → "저번에 수입지출까지 파악했었죠. 오늘은 4단계 자산부채로 넘어가겠습니다."
4. 경과 시간 표시가 있으면 자연스럽게 인사 (3일 전이면 "3일만에 다시 뵙네요")

═══════════════════════════════════
[텍스트 모드 지침]
═══════════════════════════════════
- 약간 정돈된 문장, 줄바꿈 사용 가능
- 복명복창 짧게 한 줄로
- 긴 설명/계산은 불릿 사용 가능
- 이모지 적절히 (과하지 않게, 응답당 최대 2개)
- 한 응답에 질문은 오직 1개
- 단, 5단계 오프닝의 경우: "집을 그릴 때 어디서부터" 후 고객 답변 기다림, 그다음 "재무도 똑같습니다..." 설명 후 "은퇴는 몇 세로?" 질문
- ★ 4단계 부동산은 반드시 2단계로 분리: 먼저 "자가/전세/월세?" 단독 질문 → 답변 받기 → 그 다음 "금액 얼마?" 별도 질문. 절대 같은 응답에서 두 질문을 묶지 말 것.

═══════════════════════════════════
[금지 사항]
═══════════════════════════════════
- 특정 금융회사명 절대 금지 (KB, 삼성 등)
- 특정 금융상품 추천 금지
- 단, ISA, IRP, 연금저축펀드, 펀드, ETF는 질문 시 답변 가능
- 자신의 개인 의견 추가 금지
- 한국어로만 대화
- 질문 없이 발화 종료 금지
- 환각 금지: "리포트를 보내드릴게요" 같은 없는 기능 언급 금지
- 4단계 부동산 질문을 한 응답에 묶지 말 것 (자가여부와 금액은 반드시 분리)

═══════════════════════════════════
[CFP 연결 원칙]
═══════════════════════════════════
고객이 망설이거나 복잡한 상황일 때:
"이 부분은 오상열 CFP 대표님과 직접 이야기 나누시면 훨씬 명확해질 것 같아요. 연결해 드릴까요?"

═══════════════════════════════════
[감정별 공감 표현]
═══════════════════════════════════
- 불안 → "많이 걱정되셨을 것 같아요. 충분히 이해합니다."
- 막막함 → "어디서부터 시작해야 할지 막막하셨겠어요."
- "쓰고 남으면 모아요" → "사실 그런 분들이 훨씬 많으세요."
- 퇴직/이직 → "지금 이 타이밍에 오신 게 정말 잘 오셨어요!"
- 후회 → "지금 이 순간이 가장 빠른 때입니다."

═══════════════════════════════════
[참고 기준표]
═══════════════════════════════════
비상예비자금: 맞벌이 = 월수입×3 / 외벌이 = 월수입×6
가족수별 예산 (노후연금·보험·대출 각 10% 공통):
- 1인: 생활비 20%, 저축 50%
- 2인: 생활비 30%, 저축 40%
- 3인: 생활비 40%, 저축 30%
- 4인: 생활비 50%, 저축 20%
- 5인+: 생활비 60%, 저축 10%`;

app.post('/api/consult-chat', async (req, res) => {
  try {
    const {
      message,
      userName,
      financialContext,
      conversationHistory,
      planInfo,
      resumeContext,
      systemPrompt,
      textModeInstruction
    } = req.body;

    if (!message) {
      return res.status(400).json({ success: false, error: 'message is required' });
    }

    const finalSystemPrompt = [
      systemPrompt || MONEYA_SYSTEM_PROMPT,
      textModeInstruction || '',
      planInfo || '',
      resumeContext || '',
      financialContext ? '\n[고객 정보]\n이름: ' + (financialContext.name || userName || '고객') + '\n나이: ' + (financialContext.age || '미파악') + '\n월수입: ' + (financialContext.monthlyIncome || '미파악') + '만원' : ''
    ].filter(Boolean).join('\n\n');

    const messages = [];
    if (Array.isArray(conversationHistory)) {
      conversationHistory.slice(-20).forEach(m => {
        if (m.role && m.content) {
          messages.push({
            role: m.role === 'assistant' ? 'assistant' : 'user',
            content: m.content
          });
        }
      });
    }
    messages.push({ role: 'user', content: message });

    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 400,
      system: finalSystemPrompt,
      messages: messages
    });

    const aiMessage = response.content[0]?.text || '죄송합니다. 다시 말씀해 주시겠어요?';
    
    console.log('[/api/consult-chat v3.16.5] ' + (userName || 'guest') + ': ' + message.substring(0, 30) + '... -> ' + aiMessage.substring(0, 30) + '...');
    
    res.json({ success: true, message: aiMessage });
  } catch (error) {
    console.error('[/api/consult-chat] Error:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

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
  console.log(`[OCR] 이미지 최적화 (sharp) 활성화`);
  console.log(`[음성지출] 음성 지출 입력 기능 활성화`);
  console.log(`[사전서류] Claude Vision + Excel 파싱 활성화`);
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
  let designData = null;  // 3차 금융집짓기 데이터
  let analysisContext = null;  // v3.8: OCR 분석 컨텍스트
  let spendData = null;  // v3.14: 오늘 지출 내역

  ws.on('message', (message) => {
    try {
      const msg = JSON.parse(message);

      // v3.8: OCR 분석 컨텍스트 업데이트 처리
      if (msg.type === 'update_context' && msg.analysisContext) {
        analysisContext = msg.analysisContext;
        console.log('[Realtime] OCR 분석 컨텍스트 수신:', analysisContext.fileName);
        
        // OpenAI 세션이 연결되어 있으면 프롬프트 업데이트
        if (openaiWs && openaiWs.readyState === WebSocket.OPEN) {
          const updatedPrompt = createSystemPrompt(userName, financialContext, budgetInfo, '', designData, analysisContext, spendData);
          openaiWs.send(JSON.stringify({
            type: 'session.update',
            session: {
              instructions: updatedPrompt
            }
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
        designData = msg.designData || null;  // 3차 데이터 수신
        analysisContext = msg.analysisContext || null;  // OCR 분석 컨텍스트
        spendData = msg.spendData || null;  // v3.14: 지출 내역 수신
        
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
          // 초기 세션: 1차 + 2차 + 3차 데이터 + OCR 컨텍스트 + 지출 내역 포함
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

            // 사용자 음성 텍스트 수신 시 RAG 검색
            if (event.type === 'conversation.item.input_audio_transcription.completed') {
              const userText = event.transcript;
              console.log('사용자:', userText);
              ws.send(JSON.stringify({ type: 'transcript', text: userText, role: 'user' }));
              
              // RAG 검색 수행
              const ragContext = buildRAGContext(userText);
              
              if (ragContext) {
                console.log('[Realtime] RAG 검색 결과 있음, 세션 업데이트');
                
                // ★★★ v3.14: RAG 결과 + 3차 데이터 + OCR 분석 컨텍스트 + 지출 내역 모두 포함! ★★★
                const updatedPrompt = createSystemPrompt(userName, financialContext, budgetInfo, ragContext, designData, analysisContext, spendData);
                
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
