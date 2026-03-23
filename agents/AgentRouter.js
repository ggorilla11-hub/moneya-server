const path = require('path');

// ━━━ 머니야 = 낭독기. 서버가 시키는 말만 읽는다 ━━━
const CORE = `당신은 AI재무진단 "머니야"입니다. 목소리 대역 역할만 합니다.

[절대 규칙]
① 서버가 "이 말만 하세요: TEXT" 라고 하면 TEXT만 그대로 읽는다
② TEXT 외에 단 한 글자도 추가하지 않는다
③ 자신의 의견, 설명, 질문, 공감을 절대 추가하지 않는다
④ "추가로", "혹시", "그런데", "참고로" 등 임의 추가 절대 금지
⑤ 서버 지시 없이 절대 먼저 말하지 않는다
⑥ 고객이 말해도 서버 지시 올 때까지 "네." 한 마디만 한다

[update_smart_note]
고객 답변 즉시 호출. fields 파라미터 필수.
올바른 예: note_page=1, title="인적사항", fields={"name":"오상열"}`;

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
    const filePath = path.join(__dirname, fileName);
    delete require.cache[require.resolve(filePath)];
    const fn = require(filePath);
    const result = step === 8 ? fn(subStep||1) : fn();
    return result;
  } catch(e) {
    console.error(`[AgentRouter] 로드 실패 step=${step}:`, e.message);
    return '';
  }
}

// 인라인 에이전트 스크립트 — 파일 로드 실패 시 폴백
const AGENT_SCRIPTS = {
  0: `[오프닝] "안녕하세요. 저는 AI재무진단 머니야입니다. 오상열 CFP 대표님이 20년간 직접 훈련시킨 AI에이전트입니다. 특정 금융상품이나 회사는 절대 추천하지 않으며, 순수 재무진단 목적으로만 운영됩니다. 오늘 진단에는 약 40~50분 소요됩니다. 지금 시간 괜찮으십니까?"`,
  1: `[인적사항] 서버 지시대로 이름→나이→결혼→가족→직업→맞벌이 순서로 질문한다. 서버가 지시한 말만 읽는다.`,
  2: `[경제적고민] 서버 지시대로 고민을 듣는다. 서버가 지시한 말만 읽는다.`,
  3: `[수입지출] 서버 지시대로 수입→대출→보험→연금→저축→잉여 순서로 질문한다.`,
  4: `[자산부채] 서버 지시대로 예적금→연금→투자→부동산→신용→담보 순서로 질문한다.`,
  5: `[설계도] 서버 지시대로 은퇴나이→수명 순서로 질문한다.`,
  6: `[저축투자] 서버 지시대로 포트폴리오를 안내한다.`,
  7: `[자산배분] 서버 지시대로 자산배분을 안내한다.`,
  8: `[종합설계] 서버 지시대로 7대영역을 안내한다.`,
  9: `[최종의견] 서버 지시대로 DESIRE 로드맵과 재무점수를 안내한다.`,
  10: `[클로징] 서버 지시대로 마무리한다.`
};

function buildClientSummary(d) {
  if (!d || Object.keys(d).length === 0) return '';
  const lines = [];
  if (d.name)         lines.push(`이름: ${d.name}`);
  if (d.age)          lines.push(`나이: ${d.age}`);
  if (d.marry)        lines.push(`결혼: ${d.marry}`);
  if (d.family)       lines.push(`가족: ${d.family}`);
  if (d.job)          lines.push(`직업: ${d.job}`);
  if (d.dual)         lines.push(`맞벌이: ${d.dual}`);
  if (d.w1)           lines.push(`고민: ${d.w1}`);
  if (d.income)       lines.push(`수입: ${d.income}`);
  if (d.living_cur)   lines.push(`생활비: ${d.living_cur}`);
  if (d.surplus)      lines.push(`잉여: ${d.surplus}`);
  if (d.net)          lines.push(`순자산: ${d.net}`);
  if (d.wealth_index) lines.push(`부자지수: ${d.wealth_index}`);
  if (lines.length === 0) return '';
  return `\n[현재까지 파악한 고객 정보 — 다시 묻지 않는다]\n${lines.join(' | ')}`;
}

function buildPrompt(step, subStep, session, clientData) {
  let script = '';
  try { script = loadAgent(step, subStep); } catch(e) {}
  if (!script) script = AGENT_SCRIPTS[step] || '';

  const summary = buildClientSummary(clientData);
  const prompt = `${CORE}${summary}

현재 ${session||1}회차 진단 중. 지금 단계: ${step}단계.

${script}

오원트금융연구소 | AI머니야 | 오상열 CFP`;
  console.log(`[AgentRouter] step=${step}${subStep?'.'+subStep:''} — ${prompt.length}자 (고객데이터: ${Object.keys(clientData||{}).length}개 항목)`);
  return prompt;
}

const STEP_COMPLETE_KEYS = {
  0:['session'], 1:['dual'], 2:['goal','w1'],
  3:['surplus','living_cur'], 4:['wealth_index','net'],
  5:['retire_age'], 6:['net','source'], 7:['res','inv'],
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
