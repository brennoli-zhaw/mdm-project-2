var customModelNames = null;
fetch('/getModelNames', {
    method: 'POST',
    headers: {
    },
}).then(
    response => {
        console.log(response)
        response.text().then(function (text) {
            customModelNames = JSON.parse(text);
        });

    }
).then(
    success => console.log(success)
).catch(
    error => console.log(error)
);

function checkFiles(files) {
    if (files.length != 1) {
        alert("Bitte genau eine Datei hochladen.")
        return;
    }

    const fileSize = files[0].size / 1024 / 1024; // in MiB
    if (fileSize > 10) {
        alert("Datei zu gross (max. 100Mb)");
        return;
    }

    const file = files[0];

    // Preview
    if (file) {
        preview.src = URL.createObjectURL(files[0])
    }

    // Upload
    const formData = new FormData();
    for (const name in files) {
        formData.append("image", files[name]);
    }
    //loop through all models and change element text
    let predictors = ['djl']
    predict(predictors, files, "djl")
    predict(customModelNames, files)

}

function predict(modelNames, files, path = "inference"){
    for(let i = 0; i < modelNames.length; i++){
        let model = modelNames[i];
        let answerElement = document.getElementById(model);
        //create Element if it doesn't exist already
        if(!answerElement || answerElement.length == 0){
            newElement = document.createElement("div")
            newElement.id = model;
            newElement.classList.add("card");
            answerPart.prepend(newElement)
            answerElement = document.getElementById(model);
        }
        let data = new FormData();
        for (const name in files) {
            data.append("image", files[name]);
        }
        data.append("modelName", model);
        fetch('/' + path, {
            method: 'POST',
            headers: {
            },
            body: data
        }).then(
            response => {
                if(response.status == 200){
                    response.text().then(function (text) {
                        
                        answerElement.classList.remove("error");
                        let answer = JSON.parse(text);
                        let newElements = "<ul>";
                        for (let index in answer) {
                            let object = answer[index]

                            newElements += "<li><span>" + object.className + "</span>" + object.probability + "</li>"
                            for(let key in object){
                                let value = object[key]
                                console.log(value);
                            }
                        }
                        newElements += "</ul>";
                        answerElement.innerHTML = "<h2>" + model + "</h2>" + newElements;
                    });
                } else {
                    answerElement.innerHTML = "Could not make a prediction";
                    answerElement.classList.add("error");
                }
                console.log(response)
                
            }
        ).then(
            success => console.log(success)
        ).catch(
            error => console.log(error)
        );
    }
}

