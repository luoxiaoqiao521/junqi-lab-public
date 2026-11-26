'use strict';

function visibleRanks(frame, view) {
  return view === 4 ? frame.truth : frame.views[view];
}

function reviewSeries(series, single) {
  const arrivals = [], moved = [], flags = [], exchanges = [];
  const counts = series.map((frame, ply) => {
    const change = frame.event;
    if (change) {
      const [, from, to, outcome, out, reveal] = change;
      if (out || reveal || frame.result) flags.push(ply);
      if (outcome) {
        const previous = arrivals[to];
        const response = previous !== undefined && ply - previous <= 4 ? previous : null;
        exchanges.push({ply, response, targetMoved: Boolean(moved[to])});
      }
      if (outcome !== 2) {
        moved[to] = outcome <= 1;
        arrivals[to] = outcome === 1 ? ply : undefined;
      }
      moved[from] = false;
      arrivals[from] = undefined;
      for (let n = 0; n < frame.seats.length; n++) if (frame.seats[n] < 0) {
        moved[n] = false;
        arrivals[n] = undefined;
      }
    }
    const teams = [0, 0];
    for (const seat of frame.seats) if (seat >= 0) teams[single ? seat / 2 : seat % 2]++;
    return teams;
  });
  return {counts, flags, exchanges};
}

function reviewResources(frame, nodes, view) {
  const rows = Array.from({length:4}, () => ({total:0, unknown:0, locked:0, eligible:0, engineer:0, bomb:0}));
  const seen = visibleRanks(frame, view);
  frame.seats.forEach((seat, node) => {
    if (seat < 0) return;
    const row = rows[seat], rank = seen[node];
    row.total++;
    if (rank < 0) { row.unknown++; return; }
    if (rank === 3) row.engineer++;
    if (rank === 2) row.bomb++;
    if (rank >= 2 && nodes[node][2] === 2) row.locked++;
    else if (rank >= 2) row.eligible++;
  });
  return rows.map((row, seat) => {
    const choices = !row.unknown && !frame.result && !frame.out[seat];
    return {...row, available:choices ? frame.move_origins?.[seat] : undefined,
            moves:choices ? frame.legal_move_counts?.[seat] : undefined};
  });
}

function reviewFlagRoutes(frame, view) {
  const seen = visibleRanks(frame,view);
  return frame.flag_routes?.filter(([,from,to]) => seen[from] >= 2 && seen[to] === 0);
}

let paintedReview = {data:null, limit:-1};
function drawReviewChart(data, ply, limit, single) {
  const chart = $('materialChart'), max = data.counts[0][0];
  const x = i => 30 + i / Math.max(1,data.counts.length - 1) * 278;
  const y = count => 94 - count / max * 73;
  if (paintedReview.data !== data || paintedReview.limit !== limit) {
    chart.replaceChildren();
    for (const count of [0, max]) {
      svg('line',{x1:30,y1:y(count),x2:308,y2:y(count),class:'material-grid'},chart);
      const label = svg('text',{x:25,y:y(count)+3,'text-anchor':'end',class:'material-axis'},chart);
      label.textContent = count;
    }
    for (const team of [0,1]) {
      const path = data.counts.slice(0,limit+1).map((counts,i) =>
        `${i ? `H${x(i).toFixed(2)}V` : 'M30,'}${y(counts[team]).toFixed(2)}`).join('');
      svg('path',{d:path,class:`material-line team-${team}`},chart);
    }
    for (const step of data.flags) if (step <= limit) {
      const marker = svg('circle',{cx:x(step),cy:108,r:2.7,class:'material-marker'},chart);
      marker.onclick = event => { event.stopPropagation(); move(step); };
    }
    paintedReview = {data,limit,cursor:svg('line',{class:'material-cursor',y1:15,y2:100},chart)};
  }
  paintedReview.cursor.setAttribute('x1',x(ply));
  paintedReview.cursor.setAttribute('x2',x(ply));
  const teamNames = single ? ['南方','北方'] : ['南北队','东西队'];
  $('materialLegend').textContent = teamNames.map((name,team) => `${name} ${data.counts[ply][team]} 子`).join(' · ');
  chart.onclick = event => {
    const rect = chart.getBoundingClientRect();
    const localX = (event.clientX - rect.left) * 320 / rect.width;
    move(Math.max(0,Math.min(limit,Math.round((localX-30)/278 * (data.counts.length-1)))));
  };
}

function renderResources(frame, view, single) {
  const rows = reviewResources(frame,header.nodes,view);
  const list = $('materialSeats');
  list.replaceChildren();
  rows.forEach((row,seat) => {
    if (single && seat % 2) return;
    const card = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = `${names[seat]} · ${row.total} 子${frame.out[seat] ? ' · 退场' : ''}`;
    card.appendChild(title);
    if (!frame.out[seat]) {
      const details = document.createElement('span');
      const engineer = !row.unknown && row.engineer === 1 ? '仅剩 1 工兵' : `工兵 ${row.engineer}`;
      const material = `${row.unknown ? '已见 ' : ''}${engineer}／炸弹 ${row.bomb} · 锁子 ${row.locked}`;
      const mobility = row.available !== undefined ? `能走 ${row.available}／堵住 ${row.eligible-row.available}`
        : row.moves !== undefined ? `可动资格 ${row.eligible} · ${row.moves} 个合法着`
        : row.unknown ? `暗子 ${row.unknown}` : `可动资格 ${row.eligible}`;
      details.textContent = `${material} · ${mobility}`;
      card.appendChild(details);
    }
    list.appendChild(card);
  });
  const routes = reviewFlagRoutes(frame,view);
  const current = !frame.result ? names[frame.turn] : frame.result===4 && !sourceRows ? '复盘在此截断' : '已结束';
  $('flagWatch').textContent = `下一行动：${current}。` + (frame.result ? '' : !routes ? '旧日志可补算旗口线路。'
    : routes.length ? `可见旗口接触：${routes.map(([seat,from,to]) => `${names[seat]} ${from}→${to}${seat===frame.turn?'（当前可走）':''}`).join('、')}。`
      : '当前视角没有确认的一着攻旗线路。');
}

function renderExchange(data, ply, limit, view, series) {
  const entries = data.exchanges;
  const current = entries.find(item => item.ply === ply);
  const last = entries.findLast(item => item.ply < ply);
  const next = entries.find(item => item.ply > ply && item.ply <= limit);
  $('previousExchange').onclick = () => move(last.ply);
  $('nextExchange').onclick = () => move(next.ply);
  $('previousExchange').disabled = !last;
  $('nextExchange').disabled = !next;
  $('replyExchange').hidden = current?.response == null;
  if (current?.response != null) $('replyExchange').onclick = () => move(current.response);
  if (!current) {
    $('exchangeText').textContent = '本手无碰撞。可跳到相邻交战复查。';
    return;
  }
  const before = series[ply-1], [actor,from,to,outcome] = series[ply].event;
  const seen = visibleRanks(before,view);
  const revealed = n => seen[n] < 0 ? '暗子' : ranks[seen[n]];
  const facts = `${names[actor]}${revealed(from)}对${revealed(to)}。${['','攻击方存活','防守方存活','双方同消'][outcome]}。`;
  const evidence = seen[to] < 0 && current.targetMoved ? '目标此前移动过，可排除旗和雷。' : '';
  const response = current.response == null ? '' : ` 原谱反击了第 ${current.response} 手吃子后停在这里的棋子。`;
  $('exchangeText').textContent = facts + evidence + response;
}
