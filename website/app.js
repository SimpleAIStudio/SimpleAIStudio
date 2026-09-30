const modelGrid = document.getElementById("modelGrid");
const modelCount = document.getElementById("modelCount");


async function getRegistry() {
    const locations = [
        "registry/models.json",
        "../registry/models.json"
    ];

    let lastError = null;

    for (const location of locations) {
        try {
            const response = await fetch(
                location,
                {
                    cache: "no-store"
                }
            );

            if (!response.ok) {
                throw new Error(
                    `HTTP ${response.status}`
                );
            }

            return await response.json();
        }
        catch (error) {
            lastError = error;
        }
    }

    throw lastError;
}


async function loadRegistry() {
    try {
        const registry = await getRegistry();

        const models = registry.models || [];

        modelCount.textContent =
            models.length.toString();

        renderModels(models);
    }
    catch (error) {
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


    models.forEach(
        (model, index) => {

            const card =
                document.createElement("article");


            card.className =
                "model-card";


            const parameters =
                formatParameters(
                    model.parameters
                );


            const brainSize =
                formatBytes(
                    model.brain_size_bytes
                );


            const status =
                formatStatus(
                    model.status
                );


            const modelNumber =
                String(index + 1)
                    .padStart(
                        3,
                        "0"
                    );


            const installCommand =
                model.install_command
                || `simpleai pull ${model.id}:${model.version}`;


            card.innerHTML = `

                <div class="model-top">

                    <span class="model-number">
                        MODEL ${modelNumber}
                    </span>

                    <span class="status-badge">
                        ${escapeHtml(status)}
                    </span>

                </div>


                <h3 class="model-name">
                    ${escapeHtml(model.name)}
                </h3>


                <div class="model-version">
                    ${escapeHtml(model.version)}
                </div>


                <p class="model-description">
                    ${escapeHtml(model.description)}
                </p>


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
                                model.architecture
                                || "Unknown"
                            )}
                        </span>

                    </div>


                    <div class="detail">

                        <span class="detail-label">
                            Training
                        </span>

                        <span class="detail-value">
                            ${escapeHtml(
                                model.training
                                || "Unknown"
                            )}
                        </span>

                    </div>

                </div>


                <div class="install-box">

                    <div class="install-command">
                        ${escapeHtml(
                            installCommand
                        )}
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
                card.querySelector(
                    ".copy-button"
                );


            copyButton.addEventListener(
                "click",
                () => {

                    copyText(
                        installCommand,
                        copyButton
                    );

                }
            );


            modelGrid.appendChild(
                card
            );

        }
    );

}



function formatParameters(value) {

    if (
        value === null
        || value === undefined
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
        bytes === null
        || bytes === undefined
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
    let unitIndex = 0;


    while (
        value >= 1024
        && unitIndex < units.length - 1
    ) {

        value /= 1024;

        unitIndex++;

    }


    return (
        value.toFixed(
            unitIndex === 0
                ? 0
                : 2
        )
        + " "
        + units[unitIndex]
    );

}



function formatStatus(status) {

    if (!status) {
        return "Unknown";
    }


    switch (
        status.toLowerCase()
    ) {

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

    const oldText =
        button.textContent;


    try {

        await navigator.clipboard.writeText(
            text
        );


        button.textContent =
            "COPIED";

    }

    catch {

        const textarea =
            document.createElement(
                "textarea"
            );


        textarea.value =
            text;


        document.body.appendChild(
            textarea
        );


        textarea.select();


        document.execCommand(
            "copy"
        );


        textarea.remove();


        button.textContent =
            "COPIED";

    }


    setTimeout(
        () => {

            button.textContent =
                oldText;

        },
        1200
    );

}



function escapeHtml(value) {

    const text =
        String(value ?? "");


    return text.replace(
        /[&<>"']/g,
        character => {

            const entities = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#039;"
            };


            return entities[
                character
            ];

        }
    );

}



loadRegistry();