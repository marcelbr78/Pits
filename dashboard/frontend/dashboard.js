const API_URL = 'http://localhost:8000/api';

let equityChart;

async function updateMetrics() {
    try {
        const response = await fetch(`${API_URL}/performance`);
        const data = await response.json();
        
        if (data) {
            document.getElementById('win-rate').innerText = `${(data.win_rate * 100).toFixed(1)}%`;
            document.getElementById('sharpe').innerText = data.sharpe_ratio.toFixed(2);
            document.getElementById('total-trades').innerText = data.total_trades;
        }
    } catch (e) { console.error('Error fetching metrics:', e); }
}

async function updateTrades() {
    try {
        const response = await fetch(`${API_URL}/trades`);
        const trades = await response.json();
        
        const tbody = document.querySelector('#trades-table tbody');
        tbody.innerHTML = '';
        
        let totalPnl = 0;
        const equityData = [0];

        trades.slice(-10).reverse().forEach(t => {
            const row = document.createElement('tr');
            const pnlClass = t.profit_loss >= 0 ? 'profit' : 'loss';
            row.innerHTML = `
                <td>${t.symbol}</td>
                <td>${t.side}</td>
                <td>${t.entry_price.toFixed(5)}</td>
                <td>${t.exit_price.toFixed(5)}</td>
                <td class="${pnlClass}">${(t.profit_loss * 100).toFixed(2)}%</td>
            `;
            tbody.appendChild(row);
            totalPnl += t.profit_loss;
        });

        document.getElementById('total-profit').innerText = `${(totalPnl * 100).toFixed(2)}%`;
        document.getElementById('total-profit').className = totalPnl >= 0 ? 'profit' : 'loss';

        // Update Chart
        let cumSum = 0;
        const labels = trades.map((_, i) => i);
        const data = trades.map(t => {
            cumSum += t.profit_loss;
            return cumSum * 100;
        });
        
        updateChart(labels, data);

    } catch (e) { console.error('Error fetching trades:', e); }
}

function updateChart(labels, data) {
    const ctx = document.getElementById('equityChart').getContext('2d');
    
    if (equityChart) {
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = data;
        equityChart.update();
    } else {
        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cumulative PnL (%)',
                    data: data,
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e' } },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}

// Initial Call
updateMetrics();
updateTrades();

// Poll every 5 seconds
setInterval(() => {
    updateMetrics();
    updateTrades();
}, 5000);
