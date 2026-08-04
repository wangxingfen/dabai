// 赛博公司评分系统 · 图表脚本
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var c1 = style.getPropertyValue('--c1').trim();
  var c2 = style.getPropertyValue('--c2').trim();
  var c3 = style.getPropertyValue('--c3').trim();
  var c4 = style.getPropertyValue('--c4').trim();
  var ok = style.getPropertyValue('--ok').trim();
  var warn = style.getPropertyValue('--warn').trim();
  var bad = style.getPropertyValue('--bad').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  function hexA(hex, a) {
    // 给 hex 颜色追加透明度
    if (/^#[0-9a-fA-F]{6}$/.test(hex)) {
      var r = parseInt(hex.slice(1, 3), 16);
      var g = parseInt(hex.slice(3, 5), 16);
      var b = parseInt(hex.slice(5, 7), 16);
      return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
    }
    return hex;
  }

  var baseTooltip = {
    appendToBody: true,
    backgroundColor: bg2,
    borderColor: rule,
    textStyle: { color: ink, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' }
  };
  var baseAxisLabel = { color: muted, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' };

  // ---------- Chart 1: 微笑曲线 ----------
  var smileEl = document.getElementById('chart-smile');
  if (smileEl && window.echarts) {
    var smile = echarts.init(smileEl, null, { renderer: 'svg' });
    smile.setOption({
      animation: false,
      tooltip: Object.assign({}, baseTooltip, {
        trigger: 'axis',
        formatter: function (ps) {
          var p = ps[0];
          var n = p.value[0];
          var v = p.value[1];
          var zone = n < 0.5 ? '僵化区（慢性失血）' : n <= 2.5 ? '健康区（组织焕新）' : n <= 3.5 ? '预警区（士气受损）' : '崩盘区（人才流失）';
          return '单期裁员 <b>' + n + '</b> 人<br>组织活力 <b>' + v + '</b> / 40<br><span style="color:' + muted + '">' + zone + '</span>';
        }
      }),
      grid: { left: 56, right: 48, top: 30, bottom: 44 },
      xAxis: {
        type: 'value',
        min: 0, max: 6, interval: 1,
        axisLine: { lineStyle: { color: rule } },
        axisLabel: Object.assign({}, baseAxisLabel, { formatter: '{value} 人' }),
        splitLine: { show: false },
        name: '单期裁员人数',
        nameTextStyle: { color: muted, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' },
        nameLocation: 'middle',
        nameGap: 28
      },
      yAxis: {
        type: 'value',
        min: 0, max: 40,
        axisLine: { show: false },
        axisLabel: baseAxisLabel,
        splitLine: { lineStyle: { color: rule, type: 'dashed' } },
        name: '组织活力分',
        nameTextStyle: { color: muted, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' },
        nameGap: 12
      },
      series: [{
        type: 'line',
        data: [[0, 20], [1, 40], [2, 34], [3, 22], [4, 12], [5, 5], [6, 0]],
        symbolSize: 8,
        lineStyle: { width: 3, color: accent },
        itemStyle: { color: accent, borderColor: bg2, borderWidth: 2 },
        markArea: {
          silent: true,
          itemStyle: { borderWidth: 0 },
          data: [
            [{ xAxis: 0, itemStyle: { color: hexA(warn, 0.13) }, label: { formatter: '僵化区\n不裁员', color: warn, fontSize: 12, position: 'insideTop', fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } }, { xAxis: 0.5 }],
            [{ xAxis: 0.5, itemStyle: { color: hexA(ok, 0.13) }, label: { formatter: '健康区\n裁 1-2 人', color: ok, fontSize: 12, position: 'insideTop', fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } }, { xAxis: 2.5 }],
            [{ xAxis: 2.5, itemStyle: { color: hexA(warn, 0.13) }, label: { formatter: '预警区\n裁 3 人', color: warn, fontSize: 12, position: 'insideTop', fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } }, { xAxis: 3.5 }],
            [{ xAxis: 3.5, itemStyle: { color: hexA(bad, 0.13) }, label: { formatter: '崩盘区\n裁 4 人以上', color: bad, fontSize: 12, position: 'insideTop', fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } }, { xAxis: 6 }]
          ]
        },
        markPoint: {
          symbol: 'circle',
          symbolSize: 12,
          label: {
            formatter: '最优：裁 1 人/期',
            color: ok,
            fontSize: 13,
            fontWeight: 600,
            position: 'top',
            fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif'
          },
          data: [{ coord: [1, 40], itemStyle: { color: ok, borderColor: bg2, borderWidth: 2 } }]
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: bad, type: 'dashed', width: 1.5 },
          label: { formatter: '6 人全裁 = 0 分', color: bad, fontSize: 12, position: 'end', fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' },
          data: [{ xAxis: 6 }]
        }
      }]
    });
    window.addEventListener('resize', function () { smile.resize(); });
  }

  // ---------- Chart 2: 四维权重 ----------
  var wEl = document.getElementById('chart-weights');
  if (wEl && window.echarts) {
    var weights = echarts.init(wEl, null, { renderer: 'svg' });
    weights.setOption({
      animation: false,
      tooltip: Object.assign({}, baseTooltip, {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: function (ps) {
          var p = ps[0];
          return p.name + '：<b>' + p.value + '</b> 分 / 100';
        }
      }),
      grid: { left: 90, right: 56, top: 16, bottom: 30 },
      xAxis: {
        type: 'value',
        max: 40,
        axisLine: { show: false },
        axisLabel: baseAxisLabel,
        splitLine: { lineStyle: { color: rule, type: 'dashed' } }
      },
      yAxis: {
        type: 'category',
        data: ['组织活力', '人才质量', '运营稳定', '公司声誉'],
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: Object.assign({}, baseAxisLabel, { color: ink })
      },
      series: [{
        type: 'bar',
        data: [
          { value: 40, itemStyle: { color: accent } },
          { value: 30, itemStyle: { color: hexA(accent, 0.72) } },
          { value: 20, itemStyle: { color: hexA(accent, 0.46) } },
          { value: 10, itemStyle: { color: hexA(accent, 0.28) } }
        ],
        barWidth: 26,
        label: {
          show: true,
          position: 'right',
          formatter: function (p) { return p.value + ' 分'; },
          color: ink,
          fontSize: 13,
          fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif'
        },
        itemStyle: { borderRadius: [0, 6, 6, 0] }
      }]
    });
    window.addEventListener('resize', function () { weights.resize(); });
  }

  // ---------- Chart 3: 策略对比 ----------
  var scEl = document.getElementById('chart-scenarios');
  if (scEl && window.echarts) {
    var scenarios = echarts.init(scEl, null, { renderer: 'svg' });
    scenarios.setOption({
      animation: false,
      tooltip: Object.assign({}, baseTooltip, {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: function (ps) {
          var html = ps[0].name + '<br>';
          var total = 0;
          ps.forEach(function (p) {
            html += p.marker + p.seriesName + '：<b>' + p.value + '</b><br>';
            total += p.value;
          });
          html += '总分：<b style="color:' + accent + '">' + total + '</b> / 100';
          return html;
        }
      }),
      legend: {
        data: ['组织活力', '人才质量', '运营稳定', '公司声誉'],
        textStyle: { color: muted, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' },
        top: 0
      },
      grid: { left: 48, right: 24, top: 44, bottom: 36 },
      xAxis: {
        type: 'category',
        data: ['保守派\n不裁员', '健康派\n每期 1 人', '激进派\n每期 3 人'],
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: Object.assign({}, baseAxisLabel, { color: ink, lineHeight: 18 })
      },
      yAxis: {
        type: 'value',
        max: 100,
        axisLine: { show: false },
        axisLabel: baseAxisLabel,
        splitLine: { lineStyle: { color: rule, type: 'dashed' } }
      },
      series: [
        { name: '组织活力', type: 'bar', stack: 'total', data: [14, 40, 22], barWidth: 52, itemStyle: { color: c1 }, label: { show: true, color: '#07111f', fontSize: 12, fontWeight: 600, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } },
        { name: '人才质量', type: 'bar', stack: 'total', data: [20, 20, 20], itemStyle: { color: c2 }, label: { show: true, color: '#1c0a16', fontSize: 12, fontWeight: 600, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } },
        { name: '运营稳定', type: 'bar', stack: 'total', data: [20, 20, 20], itemStyle: { color: c3 }, label: { show: true, color: '#07111f', fontSize: 12, fontWeight: 600, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } },
        { name: '公司声誉', type: 'bar', stack: 'total', data: [10, 10, 6], itemStyle: { color: c4 }, label: { show: true, color: '#1c0a16', fontSize: 12, fontWeight: 600, fontFamily: 'PingFang SC, Microsoft YaHei, sans-serif' } }
      ]
    });
    window.addEventListener('resize', function () { scenarios.resize(); });
  }
})();
