const REPO_URL =
    "https://github.com/SimpleAIStudio/SimpleAIStudio";


const REGISTRY_PATHS = [
    "registry/models.json",
    "../registry/models.json"
];


const modelsList =
    document.getElementById(
        "models-list"
    );


const modelCount =
    document.getElementById(
        "model-count"
    );


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function formatParameters(value) {
    const number =
        Number(value);


    if (!Number.isFinite(number)) {
        return "—";
    }


    if (number >= 1_000_000_000) {
        return (
            number / 1_000_000_000
        ).toFixed(2) + "B";
    }


    if (number >= 1_000_000) {
        return (
            number / 1_000_000
        ).toFixed(2) + "M";
    }


    if (number >= 1_000) {
        return (
            number / 1_000
        ).toFixed(2) + "K";
    }


    return String(number);
}


function formatBytes(value) {
    const bytes =
        Number(value);


    if (!Number.isFinite(bytes)) {
        return "—";
    }


    if (bytes >= 1024 ** 3) {
        return (
            bytes / (1024 ** 3)
        ).toFixed(2) + " GB";
    }


    if (bytes >= 1024 ** 2) {
        return (
            bytes / (1024 ** 2)
        ).toFixed(2) + " MB";
    }


    if (bytes >= 1024) {
        return (
            bytes / 1024
        ).toFixed(2) + " KB";
    }


    return bytes + " B";
}


function releasePageFromDownload(
    download
) {
    if (!download) {
        return null;
    }


    const marker =
        "/releases/download/";


    const index =
        download.indexOf(marker);


    if (index === -1) {
        return null;
    }


    const base =
        download.slice(
            0,
            index
        );


    const remaining =
        download.slice(
            index
            + marker.length
        );


    const tag =
        remaining.split("/")[0];


    if (!tag) {
        return null;
    }


    return (
        base
        + "/releases/tag/"
        + tag
    );
}


function makeMetric(
    label,
    value,
    note = ""
) {
    if (
        value === undefined
        || value === null
        || value === ""
    ) {
        return "";
    }


    return `
        <div class="metric">

            <span class="metric-label">
                ${escapeHtml(label)}
            </span>

            <span class="metric-value">
                ${escapeHtml(value)}
            </span>

            ${
                note
                    ? `
                        <span class="metric-note">
                            ${escapeHtml(note)}
                        </span>
                    `
                    : ""
            }

        </div>
    `;
}


function benchmarkMetric(model) {
    if (
        model.benchmark_accuracy
        === undefined
    ) {
        return "";
    }


    const accuracy =
        Number(
            model.benchmark_accuracy
        );


    const value =
        Number.isFinite(accuracy)
            ? accuracy.toFixed(2)
                + "%"
            : model.benchmark_accuracy;


    const note =
        model.benchmark_tests
            ? `${model.benchmark_tests} tests`
            : "";


    return makeMetric(
        "Benchmark",
        value,
        note
    );
}


function generalizationMetric(
    model
) {
    if (
        model.generalization_accuracy
        === undefined
    ) {
        return "";
    }


    const accuracy =
        Number(
            model.generalization_accuracy
        );


    const value =
        Number.isFinite(accuracy)
            ? accuracy.toFixed(2)
                + "%"
            : model.generalization_accuracy;


    const note =
        model.generalization_tests
            ? `${model.generalization_tests} tests`
            : "";


    return makeMetric(
        "Generalization",
        value,
        note
    );
}


function modelType(model) {
    if (
        model.architecture
        ?.toLowerCase()
        .includes("transformer")
    ) {
        return "Transformer Language Model";
    }


    return "Structured Reasoning Model";
}


function renderModel(
    model,
    index
) {
    const number =
        String(index + 1)
            .padStart(
                3,
                "0"
            );


    const reference =
        `${model.id}:${model.version}`;


    const pullCommand =
        model.install_command
        || `simpleai pull ${reference}`;


    const runCommand =
        `simpleai run ${reference}`;


    const releasePage =
        releasePageFromDownload(
            model.download
        );


    const skills =
        Array.isArray(
            model.skills
        )
            ? model.skills
            : [];


    const skillsHtml =
        skills.length
            ? `
                <div class="skills">

                    ${skills
                        .slice(0, 8)
                        .map(
                            skill => `
                                <span class="skill">
                                    ${escapeHtml(skill)}
                                </span>
                            `
                        )
                        .join("")}

                </div>
            `
            : "";


    const context =
        model.context
        || null;


    const tokenizer =
        model.tokenizer
        || null;


    const type =
        modelType(model);


    return `
        <article class="model-card">

            <div class="model-id">

                <div>

                    <span class="model-number">
                        MODEL ${number}
                    </span>

                    <span class="model-status">
                        ${escapeHtml(
                            model.status
                            || "released"
                        )}
                    </span>

                </div>

            </div>


            <div class="model-main">

                <div class="model-title-row">

                    <h3 class="model-title">
                        ${escapeHtml(
                            model.name
                        )}
                    </h3>

                    <span class="model-version">
                        ${escapeHtml(
                            model.version
                        )}
                    </span>

                </div>


                <p class="model-description">
                    ${escapeHtml(
                        model.description
                        || ""
                    )}
                </p>


                <div class="skills">

                    <span class="skill">
                        ${escapeHtml(type)}
                    </span>

                </div>


                ${skillsHtml}

            </div>


            <div class="model-metrics">

                ${makeMetric(
                    "Parameters",
                    formatParameters(
                        model.parameters
                    )
                )}

                ${makeMetric(
                    "Brain Size",
                    formatBytes(
                        model.brain_size_bytes
                    )
                )}

                ${makeMetric(
                    "Architecture",
                    model.architecture
                )}

                ${makeMetric(
                    "Training",
                    model.training
                )}

                ${makeMetric(
                    "Context",
                    context
                )}

                ${makeMetric(
                    "Tokenizer",
                    tokenizer
                )}

                ${benchmarkMetric(
                    model
                )}

                ${generalizationMetric(
                    model
                )}

            </div>


            <div class="model-actions">

                <div class="model-command-row">

                    <div class="model-command">

                        <code>
                            ${escapeHtml(
                                pullCommand
                            )}
                        </code>

                        <button
                            class="copy-button"
                            data-copy="${escapeHtml(
                                pullCommand
                            )}"
                            type="button"
                        >
                            COPY
                        </button>

                    </div>


                    <div class="model-command">

                        <code>
                            ${escapeHtml(
                                runCommand
                            )}
                        </code>

                        <button
                            class="copy-button"
                            data-copy="${escapeHtml(
                                runCommand
                            )}"
                            type="button"
                        >
                            COPY
                        </button>

                    </div>

                </div>


                <div class="model-links">

                    ${
                        model.download
                            ? `
                                <a
                                    class="model-link"
                                    href="${escapeHtml(
                                        model.download
                                    )}"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    Manual ZIP ↓
                                </a>
                            `
                            : ""
                    }


                    ${
                        releasePage
                            ? `
                                <a
                                    class="model-link"
                                    href="${escapeHtml(
                                        releasePage
                                    )}"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    GitHub Release ↗
                                </a>
                            `
                            : ""
                    }


                    <a
                        class="model-link"
                        href="${REPO_URL}"
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        Source ↗
                    </a>

                </div>

            </div>

        </article>
    `;
}


function normaliseRegistry(data) {
    if (Array.isArray(data)) {
        return data;
    }


    if (
        data
        && Array.isArray(
            data.models
        )
    ) {
        return data.models;
    }


    throw new Error(
        "Unknown registry format."
    );
}


async function loadRegistry() {
    let lastError = null;


    for (
        const path
        of REGISTRY_PATHS
    ) {
        try {
            const separator =
                path.includes("?")
                    ? "&"
                    : "?";


            const response =
                await fetch(
                    path
                    + separator
                    + "v="
                    + Date.now(),
                    {
                        cache:
                            "no-store"
                    }
                );


            if (!response.ok) {
                throw new Error(
                    `HTTP ${response.status}`
                );
            }


            const data =
                await response.json();


            return normaliseRegistry(
                data
            );

        } catch (error) {
            lastError =
                error;
        }
    }


    throw (
        lastError
        || new Error(
            "Could not load registry."
        )
    );
}


function bindCopyButtons() {
    const buttons =
        document.querySelectorAll(
            "[data-copy]"
        );


    for (
        const button
        of buttons
    ) {
        button.addEventListener(
            "click",
            async () => {
                const text =
                    button.dataset.copy;


                try {
                    await navigator.clipboard
                        .writeText(
                            text
                        );


                    const oldText =
                        button.textContent;


                    button.textContent =
                        "COPIED";


                    button.classList.add(
                        "copied"
                    );


                    setTimeout(
                        () => {
                            button.textContent =
                                oldText;

                            button.classList.remove(
                                "copied"
                            );
                        },
                        1200
                    );

                } catch {
                    button.textContent =
                        "FAILED";


                    setTimeout(
                        () => {
                            button.textContent =
                                "COPY";
                        },
                        1200
                    );
                }
            }
        );
    }
}


async function start() {
    try {
        const models =
            await loadRegistry();


        modelCount.textContent =
            String(
                models.length
            );


        modelsList.innerHTML =
            models
                .map(
                    renderModel
                )
                .join("");


        bindCopyButtons();

    } catch (error) {
        console.error(
            error
        );


        modelCount.textContent =
            "—";


        modelsList.innerHTML = `
            <div class="registry-error">
                Could not load the SimpleAI Registry.
            </div>
        `;


        bindCopyButtons();
    }
}


bindCopyButtons();

start();