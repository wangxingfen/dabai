/* ============================================================
 * 答题服务 —— 寻宝游戏题库接入
 *
 * 题目来源：
 *   1) 优先后端代理（server.py /api/quiz/*，由天行数据「百科题库」接口提供，
 *      答案与解析保存在服务端，只有答完才会下发校验结果）；
 *   2) 后端不可用时，回退到内置本地题库（离线兜底，本地判定答案）。
 * ============================================================ */

// 离线兜底题库（仅在后端不可用、题目拉取失败时使用）
const LOCAL_QUESTIONS = [
    { title: '下面哪个是农历五月的别称？', options: { A: '杏月', B: '桃月', C: '阳月', D: '榴月' }, answer: 'D',
      analytic: '以花命名的农历月份别称：五月——榴月，石榴红似火。' },
    { title: '二十四节气中，标志着“天气回暖、春雷始鸣”的是哪个节气？', options: { A: '立春', B: '惊蛰', C: '春分', D: '清明' }, answer: 'B',
      analytic: '惊蛰意为春雷乍动、惊醒了蛰伏的昆虫，天气回暖，万物复苏。' },
    { title: '我国最长的河流是哪一条？', options: { A: '黄河', B: '珠江', C: '长江', D: '黑龙江' }, answer: 'C',
      analytic: '长江全长约6300公里，是我国最长的河流。' },
    { title: '世界上海拔最高的山峰是？', options: { A: '乔戈里峰', B: '珠穆朗玛峰', C: '贡嘎山', D: '冈仁波齐' }, answer: 'B',
      analytic: '珠穆朗玛峰海拔8848.86米，是地球上最高的山峰。' },
    { title: '太阳系中体积最大的行星是？', options: { A: '土星', B: '海王星', C: '木星', D: '天王星' }, answer: 'C',
      analytic: '木星是太阳系中体积和质量最大的行星。' },
    { title: '光在真空中的传播速度大约是多少？', options: { A: '3万千米/秒', B: '30万千米/秒', C: '300万千米/秒', D: '3千米/秒' }, answer: 'B',
      analytic: '光在真空中的传播速度约为每秒30万千米。' },
    { title: '古诗《静夜思》的作者是谁？', options: { A: '杜甫', B: '白居易', C: '王维', D: '李白' }, answer: 'D',
      analytic: '《静夜思》（床前明月光）是唐代大诗人李白的代表作。' },
    { title: '人体最大的器官是？', options: { A: '肝脏', B: '皮肤', C: '大脑', D: '肺' }, answer: 'B',
      analytic: '皮肤覆盖全身，总面积约1.5～2平方米，是人体面积最大的器官。' },
    { title: '水的化学式是什么？', options: { A: 'H2O2', B: 'CO2', C: 'H2O', D: 'O2' }, answer: 'C',
      analytic: '水的化学式是 H₂O，由两个氢原子和一个氧原子构成。' },
    { title: '下列哪一项不属于中国的“四大发明”？', options: { A: '地动仪', B: '指南针', C: '火药', D: '印刷术' }, answer: 'A',
      analytic: '四大发明指造纸术、指南针、火药、印刷术；地动仪是张衡的发明。' },
    { title: '七大洲中面积最大的大洲是？', options: { A: '非洲', B: '北美洲', C: '亚洲', D: '南极洲' }, answer: 'C',
      analytic: '亚洲面积约4458万平方公里，是世界第一大洲。' },
    { title: '被称为“红色星球”的行星是？', options: { A: '火星', B: '金星', C: '木星', D: '水星' }, answer: 'A',
      analytic: '火星表面富含氧化铁，呈现红褐色，因此被称为“红色星球”。' },
];

export class QuizService {
    constructor() {
        this._offline = false; // 后端不可用时置 true
    }

    get offline() {
        return this._offline;
    }

    /**
     * 拉取 count 道题目（不含答案与解析）。
     * 优先后端代理；失败回退本地题库。
     */
    async fetchQuestions(count = 11) {
        // 1) 后端代理（答案隐藏在服务端）
        try {
            const res = await fetch(`/api/quiz/questions?count=${count}`, { method: 'GET' });
            if (res.ok) {
                const data = await res.json();
                const list = (data && data.questions) || [];
                if (list.length > 0) {
                    this._offline = false;
                    return list.map(q => ({ ...q, _remote: true }));
                }
            }
        } catch (e) {
            // 后端不可用，进入离线模式
        }
        // 2) 本地题库兜底（离线判定答案）
        this._offline = true;
        const shuffled = LOCAL_QUESTIONS.slice().sort(() => Math.random() - 0.5);
        return shuffled.slice(0, Math.max(1, count)).map((q, i) => ({
            id: `local_${i}_${Date.now()}`,
            _remote: false,
            title: q.title,
            options: q.options,
            answer: q.answer,
            analytic: q.analytic,
        }));
    }

    /**
     * 校验作答结果，返回 { correct, answer, analytic }。
     * 远程题目走服务端校验；本地题目直接判定。
     * 任何校验异常都明确返回「失败 + 提示」，绝不静默放行。
     */
    async checkAnswer(question, choice) {
        const letter = String(choice || '').trim().toUpperCase();
        if (!question) {
            return { correct: false, answer: letter, analytic: '（题目缺失，请重新开始游戏）' };
        }
        // 远程题目：服务端校验
        if (question._remote && !String(question.id).startsWith('local_')) {
            let data = null;
            try {
                const res = await fetch('/api/quiz/check', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id: question.id, answer: letter }),
                });
                if (res.ok) data = await res.json();
            } catch (e) {
                // 网络中断
            }
            if (data && typeof data.correct === 'boolean') {
                return {
                    correct: data.correct,
                    answer: data.answer || letter,
                    analytic: data.analytic || '（本题暂无解析）',
                };
            }
            // 网络失败或题目已失效（404）：明确告知，不静默放行
            return {
                correct: false,
                answer: letter,
                analytic: '（无法连接到题库服务器，或题目已失效，请重新开始游戏）',
            };
        }
        // 本地题目：直接判定
        const correct = letter === String(question.answer || '').trim().toUpperCase();
        return {
            correct,
            answer: question.answer || letter,
            analytic: question.analytic || '（本题暂无解析）',
        };
    }
}
