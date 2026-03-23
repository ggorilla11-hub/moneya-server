// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  ClaudeBrain — 머니야의 뇌
//  고객 발화 → Claude 판단 → 머니야 할 말 생성
//  mini Realtime은 TTS(음성출력)만 담당
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const Anthropic = require('@anthropic-ai/sdk');

const BRAIN_SYSTEM = `당신은 AI재무진단 "머니야"의 뇌입니다.
오상열 CFP 대표님의 금융집짓기® 방법론으로 훈련됐습니다.

[역할]
고객 발화를 받아서 머니야가 할 말을 JSON으로 반환합니다.
머니야는 당신이 반환한 say 값만 그대로 읽습니다.

[반환 형식 — 반드시 JSON만, 다른 말 없음]
{"say":"머니야가 할 말","field":"저장필드명","value":"정제된고객답변값","nextQ":"다음질문ID"}

[공감 공식 — 모든 발화 필수]
복명복창 + 공감 + 다음질문
예) 고객: "55세요" → say: "아, 55세이시군요. 55세에 이 진단을 받으시는 게 정말 중요한 시점이에요. 결혼은 하셨나요?"

[질문 순서]
1단계(인적사항): name→age→marry→family→job→dual
2단계(경제적고민): w1
3단계(수입지출): income→loan_cur→ins_cur→pension_cur→save_cur→surplus
4단계(자산부채): deposit→pension→invest→realty→credit→mortgage
5단계(설계도): retire_age→life_age
6단계(저축투자): inv_agree
7단계(자산배분): rebalance

[답변 정제 규칙]
- "오상열입니다" → value: "오상열"
- "55세요" → value: "55세"
- "기혼이요" → value: "기혼"
- "4인 가족이요" → value: "4인"
- "직장인이요" → value: "직장인"
- 숫자+단위: "800만원이요" → value: "800만원"

[특이사항 캐치 — 공감으로 연결]
- "남편 퇴직 예정" → "지금 이 타이밍에 오신 게 정말 잘 오셨어요!"
- "쓰고 남으면 모아요" → "그런 분들이 훨씬 많으세요."
- 불안/걱정 표현 → "많이 걱정되셨을 것 같아요. 그 마음 충분히 이해합니다."

[감정별 공감]
불안 → "많이 걱정되셨을 것 같아요."
막막함 → "어디서부터 시작해야 할지 막막하셨겠어요."
후회 → "지금 이 순간이 가장 빠른 때입니다."

[단계 전환 예고]
맞벌이 답변 후 → "지금 경제적으로 가장 큰 고민이 무엇인가요? 편하게 말씀해 주세요."
고민 답변 후 → "수입과 지출을 함께 정리해 보겠습니다. 현재 세후 한 달 실수령액이 어떻게 되세요?"
잉여자금 후 → "갖고 계신 자산과 부채를 정리해 보겠습니다. 예적금은 얼마나 있으세요?"`;

class ClaudeBrain {
  constructor(apiKey) {
    this.client = new Anthropic({ apiKey });
    this.conversationHistory = [];
    this.collectedData = {};
    this.currentStep = 1;
    this.currentQIndex = 0;
    console.log('[ClaudeBrain] ✅ 초기화 완료');
  }

  // 고객 발화 처리 → 머니야 할 말 반환
  async process(userText, currentQ) {
    const context = this._buildContext(currentQ);

    this.conversationHistory.push({
      role: 'user',
      content: `${context}\n고객 발화: "${userText}"\nJSON만 반환:`
    });

    try {
      const msg = await this.client.messages.create({
        model: 'claude-sonnet-4-5',
        max_tokens: 400,
        system: BRAIN_SYSTEM,
        messages: this.conversationHistory
      });

      const text = msg.content[0].text.trim();
      console.log(`[ClaudeBrain] 원본 응답: ${text.slice(0,100)}`);

      // JSON 파싱
      const clean = text.replace(/```json|```/g,'').trim();
      const result = JSON.parse(clean);

      // 대화 이력에 어시스턴트 응답 추가
      this.conversationHistory.push({
        role: 'assistant',
        content: text
      });

      // 고객 데이터 누적
      if (result.field && result.value) {
        this.collectedData[result.field] = result.value;
        console.log(`[ClaudeBrain] 저장: ${result.field} = "${result.value}"`);
      }

      // 대화 이력 최대 20턴 유지 (메모리 관리)
      if (this.conversationHistory.length > 40) {
        this.conversationHistory = this.conversationHistory.slice(-30);
      }

      return {
        say: result.say,
        field: result.field,
        value: result.value,
        nextQ: result.nextQ,
        success: true
      };

    } catch(e) {
      console.error('[ClaudeBrain] ❌ 오류:', e.message);
      // 폴백 — 기본 응답
      return {
        say: '네, 말씀해 주세요.',
        field: currentQ,
        value: userText,
        success: false
      };
    }
  }

  // 컨텍스트 생성
  _buildContext(currentQ) {
    const lines = [`현재 질문: ${currentQ}`];
    if (Object.keys(this.collectedData).length > 0) {
      lines.push(`지금까지 파악한 정보: ${JSON.stringify(this.collectedData)}`);
    }
    return lines.join('\n');
  }

  // 데이터 반환
  getData() { return this.collectedData; }

  // 리셋
  reset() {
    this.conversationHistory = [];
    this.collectedData = {};
    this.currentStep = 1;
    console.log('[ClaudeBrain] 리셋 완료');
  }
}

module.exports = ClaudeBrain;
