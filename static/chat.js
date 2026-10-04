const messageForm = document.getElementById("messageForm");
const textInput = document.getElementById("text");
const messageArea = document.getElementById("messageArea");
const typing = document.getElementById("typing");
const clearChat = document.getElementById("clearChat");


// ======================================================
// TIME
// ======================================================

function getTime() {

    const now = new Date();

    return now.toLocaleTimeString(
        [],
        {
            hour: "2-digit",
            minute: "2-digit"
        }
    );
}


// ======================================================
// ESCAPE HTML
// ======================================================

function escapeHTML(text) {

    const div = document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


// ======================================================
// ADD USER MESSAGE
// ======================================================

function addUserMessage(message) {

    const html = `

        <div class="message user-message">

            <div class="message-content">

                <div class="message-name">
                    You
                </div>

                <div class="bubble">
                    ${escapeHTML(message)}
                </div>

                <div class="message-time">
                    ${getTime()}
                </div>

            </div>

            <div class="message-avatar user-avatar">
                <i class="fa-solid fa-user"></i>
            </div>

        </div>

    `;

    messageArea.insertAdjacentHTML(
        "beforeend",
        html
    );

    scrollToBottom();
}


// ======================================================
// ADD BOT MESSAGE
// ======================================================

function addBotMessage(message) {

    const html = `

        <div class="message bot-message">

            <div class="message-avatar">
                ⚕
            </div>

            <div class="message-content">

                <div class="message-name">
                    MedCare AI
                </div>

                <div class="bubble">
                    ${escapeHTML(message).replace(/\n/g, "<br>")}
                </div>

                <div class="message-time">
                    ${getTime()}
                </div>

            </div>

        </div>

    `;

    messageArea.insertAdjacentHTML(
        "beforeend",
        html
    );

    scrollToBottom();
}


// ======================================================
// SCROLL
// ======================================================

function scrollToBottom() {

    messageArea.scrollTop =
        messageArea.scrollHeight;
}


// ======================================================
// SEND MESSAGE
// ======================================================

messageForm.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        const message =
            textInput.value.trim();

        if (!message) {
            return;
        }


        addUserMessage(message);

        textInput.value = "";

        textInput.disabled = true;


        // Show typing indicator

        typing.classList.remove(
            "hidden"
        );

        scrollToBottom();


        try {

            const formData =
                new FormData();

            formData.append(
                "msg",
                message
            );


            const response =
                await fetch(
                    "/get",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.text();


            typing.classList.add(
                "hidden"
            );


            addBotMessage(data);


        } catch (error) {

            console.error(error);

            typing.classList.add(
                "hidden"
            );

            addBotMessage(
                "Sorry, something went wrong. Please try again."
            );

        }


        textInput.disabled = false;

        textInput.focus();

    }
);


// ======================================================
// SUGGESTED QUESTIONS
// ======================================================

document
    .querySelectorAll(".suggestion")
    .forEach(button => {

        button.addEventListener(
            "click",
            function() {

                textInput.value =
                    this.dataset.question;

                textInput.focus();

            }
        );

    });


// ======================================================
// CLEAR CHAT
// ======================================================

clearChat.addEventListener(
    "click",
    function() {

        messageArea.innerHTML = "";

        addBotMessage(
            "Conversation cleared. How can I help you?"
        );

    }
);