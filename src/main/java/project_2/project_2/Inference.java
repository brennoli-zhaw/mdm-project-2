package project_2.project_2;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

public class Inference {

    /*
    Predictor<Image, Classifications> predictorResNet50;
    Predictor<Image, Classifications> predictorResNet18;
    Predictor<Image, Classifications> predictorCustomModel;
    */
    
    Predictor<Image, Classifications> predictor;

    public Inference(String modelName, Model model) {
        try{
            if(modelName.contains("customModel")){
                Path modelDir = Paths.get("models");
                System.out.println(model.getProperties());
                model.load(modelDir, modelName);
            }
            Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                            .addTransform(new ToTensor())
                            .optApplySoftmax(true)
                            .build();
            predictor = model.newPredictor(translator);
        } catch(Exception e){
            e.printStackTrace();
        }
    }
    

    public Classifications predictor(byte[] image) throws ModelException, TranslateException, IOException {
        InputStream is = new ByteArrayInputStream(image);
        BufferedImage bi = ImageIO.read(is);
        Image img = ImageFactory.getInstance().fromImage(bi);
        System.out.println(img);
        
        Classifications predictResult = this.predictor.predict(img);
        return predictResult;
    }
}
