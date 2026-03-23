const path = require('path');

const CORE = `당신은 AI재무진단 "머니야"입니다.
오상열 CFP 대표님의 유일한 AI 수제자입니다.

━━━ 말하기 3단계 공식 (매 발화 필수) ━━━
공감 → 복명복창 → 다음질문
예) 고객: "55세요"
머니야: "아, 55세이시군요. 55세에 이 진단을 받으시는 게 정말 중요한 시점이에요. 결혼은 하셨나요?"
★ 쌩깝(질문만 던지는 것) 절대 금지

━━━ 감정별 공감 표현 ━━━
불안 → "많이 걱정되셨을 것 같아요. 그 마음 충분히 이해합니다."
막막함 → "어디서부터 시작해야 할지 막막하셨겠어요."
후회 → "지금 이 순간이 가장 빠른 때입니다."
두려움 → "그 두려움이 오늘 이 자리에 오시게 했군요."

━━━ 절대 원칙 ━━━
• 한국어 존댓말만. 금융상품명·회사명 절대 금지
• 한 번에 질문 하나만
• 고객 답변 즉시 update_smart_note 함수 호출
• 고객이 답 안 하면 같은 질문 한 번 더. 절대 스스로 답 만들지 않음
• TV·뉴스 소리 들려도 무시하고 대기
• 단계번호·단계명 말하지 않기
• 이미 파악한 고객 정보는 절대 다시 묻지 않는다`;

const agentCache = {};

function loadAgent(step, subStep) {
  const key = `${step}_${subStep||0}`;
  const files = {
    0:'agent_00_opening', 1:'agent_01_personal', 2:'agent_02_worry',
    3:'agent_03_income',  4:'agent_04_asset',    5:'agent_05_house',
    6:'agent_06_savings', 7:'agent_07_allocation',8:'agent_08_comprehensive',
    9:'agent_09_final',   10:'agent_10_closing'
  };
  const fileName = files[step];
  if (!fileName) return '';
  try {
    // require 캐시 완전 삭제 후 로드
    const filePath = path.join(__dirname, fileName);
    delete require.cache[require.resolve(filePath)];
    const fn = require(filePath);
    const result = step === 8 ? fn(subStep||1) : fn();
    agentCache[key] = result;
    return result;
  } catch(e) {
    console.error(`[AgentRouter] 로드 실패 step=${step}:`, e.message);
    return '';
  }
}

// 지금까지 수집된 고객 데이터 요약 — 맥락 유지용
function buildClientSummary(d) {
  if (!d) return '';
  const lines = [];
  if (d.name)         lines.push(`이름: ${d.name}`);
  if (d.age)          lines.push(`나이: ${d.age}`);
  if (d.marry)        lines.push(`결혼: ${d.marry}`);
  if (d.family)       lines.push(`가족: ${d.family}`);
  if (d.job)          lines.push(`직업: ${d.job}`);
  if (d.dual)         lines.push(`맞벌이: ${d.dual}`);
  if (d.w1)           lines.push(`경제적고민: ${d.w1}`);
  if (d.goal)         lines.push(`목표: ${d.goal}`);
  if (d.income)       lines.push(`월수입: ${d.income}`);
  if (d.loan_cur)     lines.push(`대출상환: ${d.loan_cur}`);
  if (d.ins_cur)      lines.push(`보험료: ${d.ins_cur}`);
  if (d.pension_cur)  lines.push(`연금납입: ${d.pension_cur}`);
  if (d.save_cur)     lines.push(`저축투자: ${d.save_cur}`);
  if (d.living_cur)   lines.push(`생활비: ${d.living_cur}`);
  if (d.surplus)      lines.push(`잉여자금: ${d.surplus}`);
  if (d.deposit)      lines.push(`예적금: ${d.deposit}`);
  if (d.invest)       lines.push(`투자자산: ${d.invest}`);
  if (d.pension)      lines.push(`연금적립: ${d.pension}`);
  if (d.realty)       lines.push(`부동산: ${d.realty}`);
  if (d.credit)       lines.push(`신용대출: ${d.credit}`);
  if (d.mortgage)     lines.push(`담보대출: ${d.mortgage}`);
  if (d.net)          lines.push(`순자산: ${d.net}`);
  if (d.wealth_index) lines.push(`부자지수: ${d.wealth_index}`);
  if (d.retire_age)   lines.push(`은퇴나이: ${d.retire_age}`);
  if (d.life_age)     lines.push(`수명: ${d.life_age}`);
  if (lines.length === 0) return '';
  return `
━━━ 이미 파악한 고객 정보 (다시 묻지 않는다) ━━━
${lines.join(' | ')}
★ 고민 "${d.w1||''}"을 이후 모든 단계에서 대의명분으로 활용`;
}

function buildPrompt(step, subStep, session, clientData) {
  const script  = loadAgent(step, subStep);
  const summary = buildClientSummary(clientData);
  const prompt  = `${CORE}
${summary}

현재 ${session||1}회차 진단 중.

${script}

오원트금융연구소 | AI머니야 v2.1 | 오상열 CFP`;
  console.log(`[AgentRouter] step=${step}${subStep?'.'+subStep:''} — ${prompt.length}자 (고객데이터: ${Object.keys(clientData||{}).length}개 항목)`);
  return prompt;
}

const STEP_COMPLETE_KEYS = {
  0:['session'], 1:['dual'], 2:['goal','w1'],
  3:['surplus','living_cur'], 4:['wealth_index','net'],
  5:['retire_age'], 6:['net','source'], 7:['res','inv'],
  81:['monthly'], 82:['priority'], 83:['goal'],
  84:['rate'], 85:['refund'], 86:['strategy'], 87:['premium'],
  9:['score','grade'], 10:['closing']
};

function isStepComplete(notePage, subPage, fields) {
  const key = notePage===8 ? `8${subPage||1}` : notePage;
  return (STEP_COMPLETE_KEYS[key]||[]).some(k=>fields[k]&&fields[k]!=='');
}

function getNextStep(notePage, subPage) {
  if (notePage===8) {
    const next = (subPage||1)+1;
    return next<=7 ? {step:8,subStep:next} : {step:9,subStep:null};
  }
  return {step:Math.min(notePage+1,10), subStep:null};
}

module.exports = { buildPrompt, isStepComplete, getNextStep };
