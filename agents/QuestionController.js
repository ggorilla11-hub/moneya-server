// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  QuestionController — 서버가 질문을 직접 통제
//  머니야는 음성 출력만 담당
//  기존 코드 건드리지 않고 추가
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 전체 상담 질문 시나리오
const QUESTIONS = {
  // 1단계: 인적사항
  1: [
    { id:'name',    ask:'먼저 성함이 어떻게 되시나요?',        field:'name', noWait:true },
    { id:'age',     ask:'나이가 어떻게 되시나요?',              field:'age' },
    { id:'marry',   ask:'결혼은 하셨나요?',                     field:'marry' },
    { id:'family',  ask:'가족이 몇 분이세요?',                  field:'family' },
    { id:'job',     ask:'현재 어떤 일을 하고 계세요?',          field:'job' },
    { id:'dual',    ask:'맞벌이이신가요?',                      field:'dual' },
  ],
  // 2단계: 경제적 고민
  2: [
    { id:'w1', ask:'지금 경제적으로 가장 큰 고민이나 관심이 무엇인가요? 편하게 말씀해 주세요.', field:'w1' },
  ],
  // 3단계: 수입지출
  3: [
    { id:'income',      ask:'현재 세후 한 달 실수령액이 어떻게 되세요?',                    field:'income' },
    { id:'loan_cur',    ask:'현재 대출 원리금 상환 중인 것이 있으신가요? 월 얼마인가요?',    field:'loan_cur' },
    { id:'ins_cur',     ask:'보험료는 한 달에 얼마나 내고 계세요?',                          field:'ins_cur' },
    { id:'pension_cur', ask:'연금은 따로 납입하고 계신 것 있으세요? 월 얼마인가요?',         field:'pension_cur' },
    { id:'save_cur',    ask:'저축이나 투자는 한 달에 얼마 정도 하고 계세요?',                field:'save_cur' },
    { id:'surplus',     ask:'지금까지 말씀하신 것 빼고 매달 남는 돈이 있으세요? 얼마나 남으세요?', field:'surplus' },
  ],
  // 4단계: 자산부채
  4: [
    { id:'deposit',  ask:'예적금이나 청약통장은 대략 얼마나 있으세요?',          field:'deposit' },
    { id:'pension',  ask:'연금 적립금은요? 지금까지 쌓인 금액이요.',             field:'pension' },
    { id:'invest',   ask:'펀드, ETF, 주식 같은 투자 자산도 있으신가요?',         field:'invest' },
    { id:'realty',   ask:'부동산은 어떻게 되세요? 자가이신가요?',                field:'realty' },
    { id:'credit',   ask:'신용대출이 있으신가요? 있다면 얼마나 되세요?',         field:'credit' },
    { id:'mortgage', ask:'주택담보대출은요?',                                     field:'mortgage' },
  ],
  // 5단계: 금융집짓기
  5: [
    { id:'retire_age', ask:'은퇴는 몇 세로 생각하세요?',                         field:'retire_age' },
    { id:'life_age',   ask:'예상 수명은 몇 세로 보세요? 보통 90세로 잡으시면 됩니다.', field:'life_age' },
  ],
  // 6단계: 저축투자
  6: [
    { id:'inv_agree', ask:'노후연금이나 보험 부족분을 투자재원에서 빼실 건가요? 어떻게 하시겠어요?', field:'inv_agree' },
  ],
  // 7단계: 자산배분
  7: [
    { id:'rebalance', ask:'매년 정해진 날짜에 리밸런싱을 하시겠어요?', field:'rebalance' },
  ],
};

// 공감 표현 (랜덤 선택)
const EMPATHY = {
  name:        (v) => `아, ${v}님이시군요. 반갑습니다! 나이가 어떻게 되시나요?`,
  age:         (v) => `아, ${v.replace('세','')}세이시군요. ${v.replace('세','')}세에 이 진단을 받으시는 게 정말 중요한 시점이에요. 결혼은 하셨나요?`,
  marry:       (v) => v.includes('기혼')||v.includes('결혼')||v.includes('네')
                      ? '기혼이시군요. 가정을 위해 더 체계적인 계획이 필요하시겠어요.'
                      : '미혼이시군요. 지금부터 준비하시면 정말 유리하세요.',
  family:      (v) => `${v}이시군요. 가족을 위한 계획이 정말 중요하겠어요.`,
  job:         (v) => `${v}이시군요. 안정적인 수입 기반이 있으시네요.`,
  dual:        (v) => v.includes('맞벌이')||v.includes('네')
                      ? '맞벌이시군요. 두 분 소득을 합산해서 분석해 드리겠습니다.'
                      : '외벌이로 가계를 꾸려가시는군요. 더 체계적인 계획이 필요하시겠어요.',
  w1:          (v) => `${v}이시군요. 많이 걱정되셨을 것 같아요. 그 마음 충분히 이해합니다. 바로 그 문제를 해결하기 위해 오늘 진단을 하는 것입니다.`,
  income:      (v) => `월 ${v} 수령이시군요. 감사합니다.`,
  loan_cur:    (v) => v==='0'||v.includes('없')||v.includes('없음') ? '대출이 없으시군요. 정말 든든하세요.' : `월 ${v} 상환 중이시군요.`,
  ins_cur:     (v) => `보험료 월 ${v}이시군요.`,
  pension_cur: (v) => v==='0'||v.includes('없') ? '연금 납입이 없으시군요. 이 부분을 나중에 살펴보겠습니다.' : `연금 월 ${v} 납입 중이시군요.`,
  save_cur:    (v) => `저축 월 ${v}이시군요.`,
  surplus:     (v) => `잉여자금이 ${v}이시군요. 감사합니다.`,
  deposit:     (v) => `예적금 ${v}이시군요.`,
  pension:     (v) => `연금 적립금 ${v}이시군요.`,
  invest:      (v) => v==='0'||v.includes('없') ? '투자 자산이 없으시군요. 지금부터 시작하셔도 됩니다.' : `투자 자산 ${v}이시군요.`,
  realty:      (v) => v.includes('자가')||v.includes('있') ? '자가이시군요. 든든한 자산이 있으시네요.' : `${v}이시군요.`,
  credit:      (v) => v==='0'||v.includes('없') ? '신용대출이 없으시군요. 정말 잘 관리하셨어요.' : `신용대출 ${v}이시군요.`,
  mortgage:    (v) => v==='0'||v.includes('없') ? '담보대출도 없으시군요. 정말 든든하세요.' : `담보대출 ${v}이시군요.`,
  retire_age:  (v) => `${v}세 은퇴를 목표로 하시는군요.`,
  life_age:    (v) => `${v}세까지 계획하시는군요. 감사합니다.`,
};

// 단계별 예고 멘트
const STEP_INTRO = {
  1: '먼저 기본 정보를 확인하겠습니다.',
  2: '이제 경제적으로 어떤 고민이 있으신지 여쭤볼게요.',
  3: '수입과 지출을 함께 정리해 보겠습니다.',
  4: '갖고 계신 자산과 부채를 정리해 보겠습니다.',
  5: '금융집짓기 설계도를 만들어 보겠습니다.',
  6: '저축투자 포트폴리오를 말씀드리겠습니다.',
  7: '자산배분 포트폴리오를 말씀드리겠습니다.',
};

class QuestionController {
  constructor() {
    this.step = 1;          // 현재 단계
    this.qIndex = 0;        // 현재 질문 인덱스
    this.answers = {};      // 수집된 답변
    this.waitingAnswer = false; // 답변 대기 중
    this.active = false;    // 컨트롤러 활성화 여부
  }

  // 활성화
  activate() {
    this.active = true;
    this.step = 1;
    this.qIndex = 0;
    console.log('[QC] 질문 컨트롤러 활성화');
  }

  // 현재 질문 가져오기
  currentQuestion() {
    const qs = QUESTIONS[this.step];
    if (!qs || this.qIndex >= qs.length) return null;
    return qs[this.qIndex];
  }

  // 다음 질문으로 이동
  nextQuestion() {
    const qs = QUESTIONS[this.step];
    if (!qs) return null;
    this.qIndex++;
    if (this.qIndex >= qs.length) {
      // 다음 단계로
      this.step++;
      this.qIndex = 0;
      if (!QUESTIONS[this.step]) {
        console.log('[QC] 모든 질문 완료');
        return null;
      }
    }
    return this.currentQuestion();
  }

  // 답변 처리 → 공감 멘트 + 다음 질문 생성
  processAnswer(field, value, step) {
    this.answers[field] = value;
    const empathy = EMPATHY[field] ? EMPATHY[field](value) : '';
    const currentQ = this.currentQuestion();
    const nextQ = this.nextQuestion();
    
    let text = '';
    if (empathy) text += empathy;

    // 공감에 이미 다음 질문이 포함된 경우 (noWait) 추가 질문 생략
    const alreadyHasNextQ = currentQ?.noWait || empathy.includes('?');
    
    // 단계 전환 시 예고
    if (!alreadyHasNextQ && nextQ && this.qIndex === 0 && STEP_INTRO[this.step]) {
      text += ' ' + STEP_INTRO[this.step];
    }
    
    // 다음 질문 추가 (공감에 없는 경우만)
    if (!alreadyHasNextQ && nextQ) {
      text += ' ' + nextQ.ask;
    }
    
    console.log(`[QC] 답변 처리: ${field}="${value}" → 다음: ${nextQ?.id||'완료'}`);
    return { text, nextQ, done: !nextQ };
  }

  // 현재 단계의 진행 상태
  status() {
    return {
      step: this.step,
      qIndex: this.qIndex,
      current: this.currentQuestion()?.id,
      answers: Object.keys(this.answers).length
    };
  }
}

module.exports = { QuestionController, QUESTIONS, EMPATHY };
