/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package project_2.project_2;

import ai.djl.Application.CV;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * In training, multiple passes (or epochs) are made over the training data
 * trying to find patterns
 * and trends in the data, which are then stored in the model. During the
 * process, the model is
 * evaluated for accuracy using the validation data. The model is updated with
 * findings over each
 * epoch, which improves the accuracy of the model.
 */
public final class Training {

    // represents number of training samples processed before the model is updated
    private static final int BATCH_SIZE = 32;

    // the number of passes over the complete dataset
    private static final int EPOCHS = 8;

    public static void main(String[] args) throws IOException, TranslateException, ModelNotFoundException, MalformedModelException {
        // the location to save the model
        
        // create ImageFolder dataset from directory
        // ImageFolder dataset = initDataset("ut-zap50k-images-square");
        


        //Get/Train MxNetAvailable Models
        ImageFolder dataset = initDataset("raw-img-demo-tiny");
        // Split the dataset set into training dataset and validate dataset
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);
        Map<String, Model> mxNetmodels = new HashMap<String, Model>();
        //Azure can't handle all the models
        mxNetmodels.putAll(Models.trainMxNetModels(false, 0));
        for (Map.Entry<String, Model> set : mxNetmodels.entrySet()) {
            Model model = set.getValue();
            String modelName = set.getKey();
            Path modelDir = Paths.get("availableModels");
            try{
                trainModel(model, modelName, datasets, modelDir, 1);
            } catch(Exception e){
                System.out.println(e);
            }
        }

        //Get/Train Custom Models
        dataset = initDataset("raw-img-demo-demo");
        datasets = dataset.randomSplit(8, 2);
        Path modelDir = Paths.get("models");
        Map<String, Model> customModels = new HashMap<String, Model>();
        customModels.putAll(Models.trainCustomModels());
        for (Map.Entry<String, Model> set : customModels.entrySet()) {
            Model model = set.getValue();
            String modelName = set.getKey();
            try{
                trainModel(model, modelName, datasets, modelDir, 8);
            } catch(Exception e){
                System.out.println(e);
            }
        }
        
        // save labels into model directory
        Models.saveSynset(modelDir, dataset.getSynset());
    }

    private static void trainModel(Model model, String modelName, RandomAccessDataset[] datasets, Path modelDir, int epochs) throws IOException, TranslateException{
        System.out.println("");
        System.out.println("");
        System.out.println(modelName);
        // set loss function, which seeks to minimize errors
        // loss function evaluates model's predictions against the correct answer
        // (during training)
        // higher numbers are bad - means model performed poorly; indicates more errors;
        // want to
        // minimize errors (loss)
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // setting training parameters (ie hyperparameters)
        TrainingConfig config = setupTrainingConfig(loss);

        Trainer trainer = model.newTrainer(config);
        // metrics collect and report key performance indicators, like accuracy
        trainer.setMetrics(new Metrics());

        Shape inputShape = new Shape(1, 3, Models.IMAGE_HEIGHT, Models.IMAGE_HEIGHT);

        // initialize trainer with proper input shape
        trainer.initialize(inputShape);

        // find the patterns in data
        EasyTrain.fit(trainer, epochs, datasets[0], datasets[1]);

        // set model properties
        TrainingResult result = trainer.getTrainingResult();
        model.setProperty("Epoch", String.valueOf(epochs));
        model.setProperty(
                "Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

        // save the model after done training for inference later
        // model saved by model that is used
        model.save(modelDir, modelName);
        model.close();
    }

    private static ImageFolder initDataset(String datasetRoot)
        throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
            // retrieve the data
            .setRepositoryPath(Paths.get(datasetRoot))
            .optMaxDepth(3)
            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
            .addTransform(new ToTensor())
            //normalize input data according to their mean and standard deviation values. This will make different features have similar range and help our model perform better.
            .addTransform(new Normalize(new float[] {0.4914f, 0.4822f, 0.4465f}, new float[] {0.2023f, 0.1994f, 0.2010f}))
            // random sampling; don't process the data in order
            .setSampling(BATCH_SIZE, true)
            .build();

        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        //Device[] devices= new Device[]{Device.gpu(1)};
        //System.out.println(devices);
        return new DefaultTrainingConfig(loss)
            .addEvaluator(new Accuracy())
            .optDevices(Engine.getInstance().getDevices(1))
            /*.optDevices(devices)*/
            .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}
