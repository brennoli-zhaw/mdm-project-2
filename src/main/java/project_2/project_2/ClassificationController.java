package project_2.project_2;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import com.google.gson.Gson;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.repository.zoo.ModelNotFoundException;


@RestController
public class ClassificationController {
    
    /*
    private InferenceResNetImageNet inferenceResNetImageNet = new InferenceResNetImageNet();
    private InferenceResNetCifar10 inferenceResNetCifar10 = new InferenceResNetCifar10();
    private InferenceCustomModel inferenceCustomModel = new InferenceCustomModel();
    */
    private Map<String, Model> models = new HashMap<String, Model>();
    private Map<String, Inference> inferences = new HashMap<String, Inference>();

    public ClassificationController() throws ModelNotFoundException, IOException{
        try {
            //azure can't handle many models
            //models = Models.getPreTrainedMxNetBasedModels(-1);
            models.putAll(Models.trainCustomModels());
        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            e.printStackTrace();
        }
        for (Map.Entry<String, Model> set : models.entrySet()) {
            String modelName = set.getKey();
            Model model = set.getValue();
            inferences.put(modelName, new Inference(modelName, model));
        }
    }

    public Map<String, Model> getModels(){
        return models;
    }
    
    @PostMapping(path = "/inference")
    public String inference(@RequestParam("image") MultipartFile image, @RequestParam("modelName") String modelName) throws Exception {
        System.out.println("");
        System.out.println(modelName);
        return inferences.get(modelName).predictor(image.getBytes()).toJson();
    }

    @PostMapping(path = "/getModelNames")
    public String getModelNames() throws Exception {
        List<String> modelNames = new ArrayList<String>();
        try {
            for (Map.Entry<String, Model> set : models.entrySet()) {
                modelNames.add(set.getKey());
            }
        } catch(Exception e){
            System.out.println(e);
            String notWorking = "errorModel";
            return notWorking;
        }
        Gson gson = new Gson();
        String jsonArray = gson.toJson(modelNames);
        System.out.println(jsonArray);
        return jsonArray;
    }
    
    /*
    @PostMapping(path = "/inferenceResNetImageNet")
    public String inferenceResNetImageNet(@RequestParam("image") MultipartFile image) throws Exception {
        //InferenceResNet18 inferenceResNet18 = new InferenceResNet18();
        System.out.println(image);
        return inferenceResNetImageNet.predictor(image.getBytes()).toJson();
    }

    @PostMapping(path = "/inferenceResNetCifar10")
    public String predictorResNet50(@RequestParam("image") MultipartFile image) throws Exception {
        System.out.println(image);
        return inferenceResNetCifar10.predictor(image.getBytes()).toJson();
    }
    
    @PostMapping(path = "/predictorCustomModel")
    public String predictorCustomModel(@RequestParam("image") MultipartFile image) throws Exception {
        //InferenceCustomModel inferenceCustomModel = new InferenceCustomModel();
        System.out.println(image);
        return inferenceCustomModel.predictor(image.getBytes()).toJson();
    }
    */

    @SuppressWarnings("null")
    @PostMapping(path = "/djl")
    public String djlPredict(@RequestParam("image") MultipartFile image) throws Exception {
        InputStream is = new ByteArrayInputStream(image.getBytes());
        var uri = "http://localhost:8080/predictions/resnet18_v1";
        if (this.isDockerized()) {
        uri = "http://model-service:8080/predictions/resnet18_v1";
        }
        var webClient = WebClient.create();
        Resource resource = new InputStreamResource(is);
        var result = webClient.post()
            .uri(uri)
            .contentType(MediaType.MULTIPART_FORM_DATA)
            .body(BodyInserters.fromResource(resource))
            .retrieve()
            .bodyToMono(String.class)
            .block();
        return result;
    }

    private boolean isDockerized() {
    File f = new File("/.dockerenv");
    return f.exists();
    }

}