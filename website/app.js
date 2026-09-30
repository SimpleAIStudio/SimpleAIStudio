const modelGrid = document.getElementById("modelGrid");
const modelCount = document.getElementById("modelCount");


async function getRegistry() {
    const locations = [
        "registry/models.json",
        "../registry/models.json"
    ];

    let lastError;

    for (const location of locations) {
        try {
            const response = await fetch(location, {
                cache: "no-store"
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError;
}


async function loadRegistry() {
    try {
        const registry = await getRegistry();
        const models = registry.models || [];

        modelCount.textContent = models.length;
        renderModels(models);
    } catch (error) {
        console.error(
            "Could not load SimpleAI registry:",
            error
        );

        modelGrid.innerHTML = `
            <div class="error-message">
                Could not load the SimpleAI model registry.
            </div>
        `;
    }
}


function renderModels(models) {
    modelGrid.innerHTML = "";

    if (models.length === 0) {
        modelGrid.innerHTML = `
            <div class="empty-message">
                No models have been published yet.
            </div>
        `;

        return;
    }

    models.forEach((model, index) => {
        const card = document.createElement("article");

        card.className = "model-card";

        const number = String(index + 1).padStart(3, "0");

        const parameters = formatParameters(
            model.parameters
        );

        const brainSize = formatBytes(
            model.brain_size_bytes
        );

        const status = formatStatus(
            model.status
        );

        const installCommand =
            model.install_command ||
            `simpleai pull ${model.id}:${model.version}`;

        const benchmark = makeMetric(
            "Benchmark",
            model.benchmark_accuracy,
            model.benchmark_scope
        );

        const generalization = makeMetric(
            "Generalization",
            model.generalization_accuracy,
            model.generalization_scope
        );

        card.innerHTML = `
            <div class="model-top">
                <span class="model-number">
                    MODEL ${number}
                </span>

                <span class="status-badge">
                    ${escapeHtml(status)}
                </span>
            </div>

            <div class="model-main">
                <div class="model-heading">
                    <h3 class="model-name">
                        ${escapeHtml(model.name)}
                    </h3>

                    <span class="model-version">
                        ${escapeHtml(model.version)}
                    </span>
                </div>

                <p class="model-description">
                    ${escapeHtml(model.description)}
                </p>
            </div>

            <div class="model-details">
                <div class="detail">
                    <span class="detail-label">
                        Parameters
                    </span>

                    <span class="detail-value">
                        ${escapeHtml(parameters)}
                    </span>
                </div>

                <div class="detail">
                    <span class="detail-label">
                        Brain Size
                    </span>

                    <span class="detail-value">
                        ${escapeHtml(brainSize)}
                    </span>
                </div>

                <div class="detail">
                    <span class="detail-label">
                        Architecture
                    </span>

                    <span class="detail-value">
                        ${escapeHtml(
                            model.architecture || "Unknown"
                        )}
                    </span>
                </div>

                <div class="detail">
                    <span class="detail-label">
                        Training
                    </span>

                    <span class="detail-value">
                        ${escapeHtml(
                            model.training || "Unknown"
                        )}
                    </span>
                </div>

                ${benchmark}
                ${generalization}
            </div>

            <div class="install-box">
                <div class="install-command">
                    ${escapeHtml(installCommand)}
                </div>

                <button
                    class="copy-button"
                    type="button"
                >
                    COPY
                </button>
            </div>
        `;

        const copyButton =
            card.querySelector(".copy-button");

        copyButton.addEventListener(
            "click",
            () => {
                copyText(
                    installCommand,
                    copyButton
                );
            }
        );

        modelGrid.appendChild(card);
    });
}


function makeMetric(
    label,
    value,
    scope
) {
    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    const percent =
        Number(value).toFixed(2) + "%";

    const title = scope
        ? ` title="${escapeHtml(scope)}"`
        : "";

    return `
        <div class="detail metric"${title}>
            <span class="detail-label">
                ${escapeHtml(label)}
            </span>

            <span class="detail-value metric-value">
                ${escapeHtml(percent)}
            </span>
        </div>
    `;
}


function formatParameters(value) {
    if (
        value === null ||
        value === undefined
    ) {
        return "TBD";
    }

    if (value >= 1_000_000_000) {
        return (
            value / 1_000_000_000
        ).toFixed(2) + "B";
    }

    if (value >= 1_000_000) {
        return (
            value / 1_000_000
        ).toFixed(2) + "M";
    }

    if (value >= 1_000) {
        return (
            value / 1_000
        ).toFixed(2) + "K";
    }

    return value.toLocaleString();
}


function formatBytes(bytes) {
    if (
        bytes === null ||
        bytes === undefined
    ) {
        return "TBD";
    }

    if (bytes === 0) {
        return "0 B";
    }

    const units = [
        "B",
        "KB",
        "MB",
        "GB"
    ];

    let value = bytes;
    let unit = 0;

    while (
        value >= 1024 &&
        unit < units.length - 1
    ) {
        value /= 1024;
        unit++;
    }

    const decimals =
        unit === 0 ? 0 : 2;

    return (
        value.toFixed(decimals) +
        " " +
        units[unit]
    );
}


function formatStatus(status) {
    if (!status) {
        return "Unknown";
    }

    switch (status.toLowerCase()) {
        case "development":
            return "In Development";

        case "released":
            return "Released";

        case "experimental":
            return "Experimental";

        case "deprecated":
            return "Deprecated";

        default:
            return status;
    }
}


async function copyText(
    text,
    button
) {
    const oldText = button.textContent;

    try {
        await navigator.clipboard.writeText(
            text
        );
    } catch {
        const textarea =
            document.createElement("textarea");

        textarea.value = text;

        document.body.appendChild(
            textarea
        );

        textarea.select();

        document.execCommand("copy");

        textarea.remove();
    }

    button.textContent = "COPIED";

    setTimeout(
        () => {
            button.textContent = oldText;
        },
        1200
    );
}


function escapeHtml(value) {
    const text = String(
        value ?? ""
    );

    const entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    };

    return text.replace(
        /[&<>"']/g,
        character => entities[character]
    );
}


loadRegistry();