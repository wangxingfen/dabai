/* ============================================================
 * 答题弹窗 UI —— 寻宝游戏
 *
 * 流程：展示题目与四个选项 → 玩家作答 → 校验 → 展示答案与解析
 * （答案和解析只在作答完成后出现）→ 点击「继续」返回游戏。
 *
 * present() 返回 Promise，在玩家点「继续」后 resolve 为
 * { correct, answer, analytic, choice }。
 * ============================================================ */

const KIND_LABELS = {
    star: '✨ 星光谜题',
    clue: '🔍 线索谜题',
    treasure: '🏆 宝藏谜题',
};

const RESULT_AUTO_CLOSE_MS = 20000; // 结果解析最多展示 20 秒后自动关闭

export class QuizUI {
    constructor() {
        this._overlay = null;
        this._busy = false;
        this._resultTimer = null;   // 结果自动关闭定时器
        this._resultCountdown = null; // 倒计时刷新句柄
    }

    get isOpen() {
        return !!this._overlay;
    }

    /**
     * 展示一道题，等待玩家作答并确认结果。
     * @param {Object} opts
     *   index      第几题
     *   total      总题数
     *   kind       'star' | 'clue' | 'treasure'
     *   question   { title, options: {A,B,C,D} }
     *   checkAnswer async (letter) => { correct, answer, analytic }
     * @returns {Promise<{correct, answer, analytic, choice}>}
     */
    present(opts) {
        return new Promise(resolve => {
            // 先渲染题目（_renderQuestion 内部会 _clear → close 清理上一轮状态），
            // 之后再赋值 _resolve / _opts，否则 close() 会把它们置空，
            // 导致"继续"按钮点击时 resolve 为 null、Promise 永不 resolve。
            this._renderQuestion(opts);
            this._resolve = resolve;
            this._opts = opts;
        });
    }

    close() {
        // 清理结果自动关闭定时器
        if (this._resultTimer) { clearTimeout(this._resultTimer); this._resultTimer = null; }
        if (this._resultCountdown) { clearInterval(this._resultCountdown); this._resultCountdown = null; }
        if (this._overlay && this._overlay.parentNode) {
            this._overlay.parentNode.removeChild(this._overlay);
        }
        this._overlay = null;
        this._resolve = null;
        this._opts = null;
        this._busy = false;
    }

    // ---------- 题目视图 ----------
    _renderQuestion(opts) {
        this._clear();
        const kindLabel = KIND_LABELS[opts.kind] || '📜 谜题';
        const overlay = document.createElement('div');
        overlay.className = 'quiz-overlay';

        const panel = document.createElement('div');
        panel.className = 'quiz-panel';

        const header = document.createElement('div');
        header.className = 'quiz-header';
        const kind = document.createElement('span');
        kind.className = 'quiz-kind';
        kind.textContent = kindLabel;
        const progress = document.createElement('span');
        progress.className = 'quiz-progress';
        progress.textContent = `第 ${opts.index}/${opts.total} 题`;
        header.appendChild(kind);
        header.appendChild(progress);

        const title = document.createElement('div');
        title.className = 'quiz-title';
        title.textContent = opts.question.title;

        const optionsBox = document.createElement('div');
        optionsBox.className = 'quiz-options';
        const letters = ['A', 'B', 'C', 'D'];
        letters.forEach(letter => {
            const btn = document.createElement('button');
            btn.className = 'quiz-option';
            btn.dataset.letter = letter;
            const letterEl = document.createElement('span');
            letterEl.className = 'quiz-letter';
            letterEl.textContent = letter;
            const textEl = document.createElement('span');
            textEl.className = 'quiz-option-text';
            textEl.textContent = opts.question.options[letter] || '（无此选项）';
            btn.appendChild(letterEl);
            btn.appendChild(textEl);
            btn.addEventListener('click', () => this._onChoose(letter, opts, optionsBox));
            optionsBox.appendChild(btn);
        });

        panel.appendChild(header);
        panel.appendChild(title);
        panel.appendChild(optionsBox);
        overlay.appendChild(panel);
        document.body.appendChild(overlay);
        this._overlay = overlay;
    }

    async _onChoose(letter, opts, optionsBox) {
        if (this._busy) return;
        this._busy = true;

        // 锁定选项，标出所选
        const buttons = optionsBox.querySelectorAll('.quiz-option');
        buttons.forEach(b => {
            b.classList.add('disabled');
            if (b.dataset.letter === letter) b.classList.add('chosen');
        });
        const chosenBtn = optionsBox.querySelector(`.quiz-option[data-letter="${letter}"]`);

        // 服务端/本地校验（答案与解析只在作答完成后返回）。
        // 任何校验异常都必须正常渲染结果，避免弹窗卡死、收集/结算无法继续。
        let result;
        try {
            result = await opts.checkAnswer(letter);
        } catch (e) {
            result = { correct: false, answer: letter, analytic: '（校验服务异常，请重试）' };
        }
        if (!result || typeof result.correct !== 'boolean') {
            result = { correct: false, answer: letter, analytic: '（校验失败，请重试）' };
        }
        const correct = !!result.correct;

        // 标记正确/错误
        buttons.forEach(b => {
            if (b.dataset.letter === result.answer) b.classList.add('correct');
        });
        if (!correct && chosenBtn) chosenBtn.classList.add('wrong');
        if (correct && chosenBtn) chosenBtn.classList.remove('chosen');

        // 展示结果（答案与解析此时才出现）
        setTimeout(() => this._renderResult({ ...result, choice: letter }), 450);
    }

    // ---------- 结果视图（答完才出现答案与解析） ----------
    _renderResult(result) {
        if (!this._overlay) return;
        const panel = this._overlay.querySelector('.quiz-panel');
        if (!panel) return;

        const box = document.createElement('div');
        box.className = 'quiz-result';

        const icon = document.createElement('div');
        icon.className = 'quiz-result-icon';
        icon.textContent = result.correct ? '✅' : '❌';

        const title = document.createElement('div');
        title.className = `quiz-result-title ${result.correct ? 'ok' : 'no'}`;
        title.textContent = result.correct ? '回答正确！' : '回答错误…';

        const opts = this._opts || {};
        const answerLine = document.createElement('div');
        answerLine.className = 'quiz-result-answer';
        const answerText = opts.question && opts.question.options[result.answer]
            ? `${result.answer}. ${opts.question.options[result.answer]}`
            : result.answer;
        answerLine.textContent = `正确答案：${answerText}`;

        const analytic = document.createElement('div');
        analytic.className = 'quiz-result-analytic';
        analytic.textContent = result.analytic ? `📖 ${result.analytic}` : '📖 （本题暂无解析）';

        // 倒计时提示（每秒刷新，到 0 自动关闭）
        const countdownEl = document.createElement('div');
        countdownEl.className = 'quiz-result-countdown';

        const nextBtn = document.createElement('button');
        nextBtn.className = 'quiz-next-btn';
        const btnLabel = result.correct
            ? (opts.index >= opts.total ? '🎉 冲向宝藏！' : '继续冒险 →')
            : '😢 挑战失败';

        // 统一的关闭逻辑：resolve 结果并清理（定时器/弹窗）
        const finish = () => {
            const resolve = this._resolve;
            this.close();
            if (resolve) resolve(result);
        };

        nextBtn.textContent = btnLabel;
        nextBtn.addEventListener('click', () => {
            if (this._resultTimer) { clearTimeout(this._resultTimer); this._resultTimer = null; }
            if (this._resultCountdown) { clearInterval(this._resultCountdown); this._resultCountdown = null; }
            finish();
        });

        box.appendChild(icon);
        box.appendChild(title);
        box.appendChild(answerLine);
        if (analytic.textContent) box.appendChild(analytic);
        box.appendChild(nextBtn);
        box.appendChild(countdownEl);

        // 替换面板内容为结果视图
        panel.innerHTML = '';
        panel.appendChild(box);

        // 启动 20 秒自动关闭倒计时
        let remain = Math.ceil(RESULT_AUTO_CLOSE_MS / 1000);
        countdownEl.textContent = `⏱ ${remain}s 后自动继续`;
        this._resultCountdown = setInterval(() => {
            remain--;
            if (remain <= 0) {
                clearInterval(this._resultCountdown);
                this._resultCountdown = null;
                return;
            }
            countdownEl.textContent = `⏱ ${remain}s 后自动继续`;
        }, 1000);
        this._resultTimer = setTimeout(() => {
            this._resultTimer = null;
            this._resultCountdown = null;
            // 仅在弹窗仍然打开时自动关闭（玩家可能已手动点了继续）
            if (this._overlay) finish();
        }, RESULT_AUTO_CLOSE_MS);
    }

    _clear() {
        this.close();
    }
}
