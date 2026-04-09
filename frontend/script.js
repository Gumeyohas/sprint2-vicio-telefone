// ── Configuração ────────────────────────────────────────────────
const API_URL = 'http://localhost:5000/predict';

// ── IDs dos campos do formulário ─────────────────────────────────
const CAMPOS = [
  'age',
  'gender',
  'daily_screen_time_hours',
  'social_media_hours',
  'gaming_hours',
  'work_study_hours',
  'sleep_hours',
  'notifications_per_day',
  'app_opens_per_day',
  'weekend_screen_time',
  'stress_level',
  'academic_work_impact',
];

// ── Coleta os dados do formulário ────────────────────────────────
function coletarDados() {
  const dados = {};
  CAMPOS.forEach((campo) => {
    const elemento = document.getElementById(campo);
    dados[campo] = parseFloat(elemento.value);
  });
  return dados;
}

// ── Chama a API Flask ────────────────────────────────────────────
async function chamarAPI(dados) {
  const resposta = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dados),
  });

  if (!resposta.ok) {
    throw new Error(`Erro na API: ${resposta.status}`);
  }

  return resposta.json();
}

// ── Renderiza o resultado na tela ────────────────────────────────
function exibirResultado(resultado) {
  const card        = document.getElementById('result-card');
  const conteudo    = document.getElementById('result-content');
  const isViciado   = resultado.preocupante === 1;

  const probViciado    = resultado.prob_preocupante;
  const probNaoViciado = resultado.prob_equilibrado;

  conteudo.innerHTML = `
    <div class="verdict ${isViciado ? 'verdict--addicted' : 'verdict--safe'}">
      <div class="verdict-icon">${isViciado ? '🔴' : '🟢'}</div>
      <div>
        <div class="verdict-label">Resultado da classificação</div>
        <div class="verdict-title">${isViciado ? 'Padrão de Uso Preocupante' : 'Uso Digital Equilibrado'}</div>
      </div>
    </div>

    <div class="prob-row">
      <div class="prob-item">
        <div class="prob-header">
          <span>Probabilidade — Uso Preocupante</span>
          <strong>${probViciado.toFixed(2)}%</strong>
        </div>
        <div class="prob-track">
          <div class="prob-fill prob-fill--danger" id="fill-danger"></div>
        </div>
      </div>

      <div class="prob-item">
        <div class="prob-header">
          <span>Probabilidade — Uso Equilibrado</span>
          <strong>${probNaoViciado.toFixed(2)}%</strong>
        </div>
        <div class="prob-track">
          <div class="prob-fill prob-fill--success" id="fill-success"></div>
        </div>
      </div>
    </div>

    <div class="model-info">
      <span class="chip">Algoritmo: SVM (kernel=rbf)</span>
      <span class="chip">ROC-AUC: 0.974</span>
      <span class="chip">Accuracy: 92.33%</span>
      <span class="chip">F1-Score: 0.947</span>
    </div>
  `;

  // Mostra o card
  card.classList.add('visible');
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Anima as barras de probabilidade com pequeno delay
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('fill-danger').style.width  = `${probViciado}%`;
      document.getElementById('fill-success').style.width = `${probNaoViciado}%`;
    }, 100);
  });
}

// ── Exibe mensagem de erro ────────────────────────────────────────
function exibirErro(mensagem) {
  const card     = document.getElementById('result-card');
  const conteudo = document.getElementById('result-content');

  conteudo.innerHTML = `
    <div class="verdict verdict--addicted">
      <div class="verdict-icon">⚠️</div>
      <div>
        <div class="verdict-label">Erro na requisição</div>
        <div class="verdict-title" style="font-size:1.1rem">${mensagem}</div>
      </div>
    </div>
  `;

  card.classList.add('visible');
}

// ── Handler principal do botão ────────────────────────────────────
async function analisarPerfil() {
  const btn      = document.getElementById('btn-analyze');
  const btnText  = btn.querySelector('.btn-text');
  const btnIcon  = btn.querySelector('.btn-icon');

  // Estado de loading
  btn.classList.add('loading');
  btn.disabled    = true;
  btnText.textContent = 'Analisando';
  btnIcon.style.display = 'none';

  try {
    const dados     = coletarDados();
    const resultado = await chamarAPI(dados);
    exibirResultado(resultado);
  } catch (erro) {
    console.error(erro);
    exibirErro('Não foi possível conectar ao servidor. Verifique se o back-end está rodando.');
  } finally {
    // Restaura o botão
    btn.classList.remove('loading');
    btn.disabled            = false;
    btnText.textContent     = 'Analisar Perfil';
    btnIcon.style.display   = '';
  }
}

// ── Event Listeners ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-analyze').addEventListener('click', analisarPerfil);
});