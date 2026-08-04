// 蜂群游戏引擎报告图表
(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var cjkFont = "'InstrumentSans','Noto Sans CJK SC','WenQuanYi Micro Hei','Microsoft YaHei',sans-serif";

  // --- Chart: 图 5-2 推理流量分层（环形图） ---
  var tier = echarts.init(document.getElementById('chart-tier'), null, { renderer: 'svg' });
  tier.setOption({
    textStyle: { fontFamily: cjkFont },
    tooltip: {
      trigger: 'item',
      appendToBody: true,
      backgroundColor: bg2,
      borderColor: rule,
      textStyle: { color: ink },
      formatter: '{b}：{d}%'
    },
    legend: { bottom: 0, textStyle: { color: muted }, itemWidth: 14, itemHeight: 14 },
    animation: false,
    color: [accent, accent2, accent + '55'],
    series: [{
      type: 'pie',
      radius: ['42%', '68%'],
      center: ['50%', '44%'],
      label: { color: ink, formatter: '{b}\n{d}%' },
      labelLine: { lineStyle: { color: rule } },
      itemStyle: { borderColor: '#0a0e16', borderWidth: 2 },
      data: [
        { value: 70, name: '小模型（常规决策）' },
        { value: 20, name: '大模型（复杂规划）' },
        { value: 10, name: '缓存命中 / 模板化' }
      ]
    }]
  });
  window.addEventListener('resize', function() { tier.resize(); });

  // --- Chart: 图 7-1 容量模型对比（分组条形图） ---
  var cap = echarts.init(document.getElementById('chart-capacity'), null, { renderer: 'svg' });
  cap.setOption({
    textStyle: { fontFamily: cjkFont },
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      backgroundColor: bg2,
      borderColor: rule,
      textStyle: { color: ink },
      axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(148,163,184,0.08)' } }
    },
    legend: { top: 0, textStyle: { color: muted }, itemWidth: 14, itemHeight: 14 },
    animation: false,
    color: [accent, accent2, accent + '55'],
    grid: { left: 56, right: 16, top: 46, bottom: 8 },
    xAxis: {
      type: 'category',
      data: ['在线 Agent\n（万）', '决策请求\n（千/s）', '解码吞吐\n（M tok/s）', 'GPU 估算\n（H200 当量）'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, interval: 0, lineHeight: 16 },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      { name: '5k 并发世界', type: 'bar', barGap: '8%', data: [7.5, 1.9, 0.28, 50], itemStyle: { borderRadius: [4, 4, 0, 0] } },
      { name: '20k 并发世界', type: 'bar', data: [30, 7.5, 1.13, 200], itemStyle: { borderRadius: [4, 4, 0, 0] } },
      { name: '50k 并发世界', type: 'bar', data: [75, 18.8, 2.81, 500], itemStyle: { borderRadius: [4, 4, 0, 0] } }
    ]
  });
  window.addEventListener('resize', function() { cap.resize(); });
})();
