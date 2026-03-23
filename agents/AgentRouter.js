// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  AgentRouter — 단계별 에이전트 라우터
//  서버 메모리: 핵심 원칙 800자만 상주
//  각 에이전트: 단계 진입 시 동적 로드
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const path = require('path');

// 핵심 원칙 — 항상 포함 (~700자)
const CORE = `당신은 AI재무진단 "머니야"입니다.
오상열 CFP 대표님의 유일한 AI 수제자입니다.

━━━ 말하기 공식 (모든 발화 필수) ━━━
공감 → 복명복창 → 다음질문 순서로 반드시 말한다.
예) 고객: "55세요" → 머니야: "아, 55세이시군요.(공감+복명복창) 55세에 이 진단을 받으시는 게 정말 중요한 시점이에요.(공감 추가) 결혼은 하셨나요?(다음질문)"
절대 질문만 던지는 쌩깝 금지.

━━━ 절대 원칙 ━━━
• 한국어 존댓말만. 금융상품명·회사명 금지
• 한 번에 질문 하나만
• 고객 답변 즉시 update_smart_note 함수 호출 (텍스트로 말하면 안 됨)
• 고객이 답 안 하면 같은 질문 한 번 더. 절대 스스로 답 만들지 않음
• TV·뉴스 소리 들려도 무시하고 대기
• update_smart_note를 텍스트로 "(update_smart_note...)" 출력하는 것 절대 금지

━━━ 수정 처리 ━━━
"아니요" "틀렸어요" → "네, 말씀하세요." → 수정 확인 후 노트 업데이트`;

// 에이전트 로드 캐시 (최초 1회만 로드)
const agentCache = {};

function loadAgent(step, subStep) {
  const key = `${step}_${subStep||0}`;
  if (agentCache[key]) return agentCache[key];

  const agentFiles = {
    0:  'agent_00_opening',
    1:  'agent_01_personal',
    2:  'agent_02_worry',
    3:  'agent_03_income',
    4:  'agent_04_asset',
    5:  'agent_05_house',
    6:  'agent_06_savings',
    7:  'agent_07_allocation',
    8:  'agent_08_comprehensive',
    9:  'agent_09_final',
    10: 'agent_10_closing',
  };

  const fileName = agentFiles[step];
  if (!fileName) return '';

  try {
    const agentFn = require(path.join(__dirname, fileName));
    const result = step === 8 ? agentFn(subStep || 1) : agentFn();
    agentCache[key] = result;
    return result;
  } catch (e) {
    console.error(`[에이전트] 로드 실패 step=${step}:`, e.message);
    return '';
  }
}

// 현재 단계 프롬프트 생성 — 핵심원칙 + 현재단계 에이전트
function buildPrompt(step, subStep, session) {
  const agentScript = loadAgent(step, subStep);
  const sessionInfo = `현재 ${session||1}회차 ${session===1?'초회진단':'정기진단'}입니다.`;

  const prompt = `${CORE}

${sessionInfo}

━━━ update_smart_note 원칙 ━━━
고객 답변 즉시 호출. 모아서 한 번에 하지 않음. 빈값·플레이스홀더 금지.

${agentScript}

오원트금융연구소 | AI머니야 v6.7 멀티에이전트 | 오상열 CFP`;

  console.log(`[AgentRouter] step=${step}${subStep?'.'+subStep:''} 프롬프트 빌드 완료 — ${prompt.length}자`);
  return prompt;
}

// 단계 완료 키 — 이 필드가 채워지면 다음 단계로 전환
const STEP_COMPLETE_KEYS = {
  0:  ['session', 'disclaimer'],
  1:  ['dual'],
  2:  ['goal', 'w1'],
  3:  ['surplus', 'living_cur'],
  4:  ['wealth_index', 'net'],
  5:  ['desire', 'strategy', 'retire_age'],
  6:  ['net', 'source'],
  7:  ['res', 'inv'],
  81: ['monthly'],
  82: ['priority'],
  83: ['goal'],
  84: ['rate'],
  85: ['refund'],
  86: ['strategy'],
  87: ['premium'],
  9:  ['score', 'grade'],
  10: ['closing'],
};

function isStepComplete(notePage, subPage, fields) {
  const key = notePage === 8 ? `8${subPage||1}` : notePage;
  const completeKeys = STEP_COMPLETE_KEYS[key] || [];
  return completeKeys.some(k => fields[k] && fields[k] !== '');
}

function getNextStep(notePage, subPage) {
  if (notePage === 8) {
    const nextSub = (subPage || 1) + 1;
    if (nextSub <= 7) return { step: 8, subStep: nextSub };
    return { step: 9, subStep: null };
  }
  return { step: Math.min(notePage + 1, 10), subStep: null };
}

module.exports = { buildPrompt, isStepComplete, getNextStep, CORE };
