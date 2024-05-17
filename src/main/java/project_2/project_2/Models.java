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

/*
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
*/

import ai.djl.*;
import ai.djl.Application.CV;
//import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.ndarray.types.*;
import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.*;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.*;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** A helper class loads and saves model. */
public final class Models {

    // the number of classification labels: boots, sandals, shoes, slippers
    public static final int NUM_OF_OUTPUT = 10;

    // the height and width for pre-processing of the image
    public static final int IMAGE_HEIGHT = 50;
    public static final int IMAGE_WIDTH = 50;
    //available imageClassifications of mxnet can be found here: https://github.com/deepjavalibrary/djl/blob/master/engines/mxnet/mxnet-model-zoo/src/test/resources/mlrepo/model/cv/image_classification/ai/djl/mxnet/resnet/metadata.json
    public static final String[] CUSTOM_MODELS = {
        "customModelDemi",
        "customModelTall",
        "customModelGrande",
        "customModelVenti"
    };

    private Models() {}

    public static Set<String> listFilesUsingJavaIO(String dir) {
        return Stream.of(new File(dir).listFiles())
        .filter(file -> !file.isDirectory())
        .map(File::getName)
        .collect(Collectors.toSet());
    }

    public static Map<String, Model> getPreTrainedMxNetBasedModels(int limit) throws ModelNotFoundException, MalformedModelException, IOException {
        List<String> modelNames = new ArrayList<String>();
        Map<String, Model> models = new HashMap<String, Model>();
        List<Artifact> imageClassificationModelList = MxModelZoo.listModels().get(CV.IMAGE_CLASSIFICATION);
        int i = 0;
        Set<String> availableModels = listFilesUsingJavaIO("availableModels");
        //System.out.println(trainedModels);
        String fileEnding = "-0001.params";
        for(Artifact artifact : imageClassificationModelList) {
            if(i > limit && limit != -1){
                continue;
            }
            i++;
            //Map<String, String> filters = new HashMap<>();
            String artifactId = artifact.getMetadata().getArtifactId();
            List<Artifact> subArtifacts = artifact.getMetadata().getArtifacts();
            for(Artifact subArtifact : subArtifacts){
                try{
                    Map<String, String> filters = subArtifact.getProperties();
                    String modelName = subArtifact.getName();
                    for (Map.Entry<String, String> set :
                        filters.entrySet()) {
                        // Printing all elements of a Map
                        modelName = modelName + set.getKey() + "_" + set.getValue() + "_";
                    }
                    //no duplicates
                    if(modelNames.isEmpty() || modelNames.contains(modelName) != true){
                        modelNames.add(modelName);
                    } else{
                        continue;
                    }
                    String modelFile = modelName + fileEnding;
                    if(!availableModels.contains(modelFile)){
                        continue;
                    }
                    
                    System.out.println(modelName);
                    
                    Criteria<Image, Classifications> criteria = Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId(artifactId)
                        .optFilters(filters)
                        .build();
                    Model model = criteria.loadModel();
                    
                    SequentialBlock newBlock = new SequentialBlock();
                    SymbolBlock block = (SymbolBlock) model.getBlock();
                    block.removeLastBlock();
                    newBlock.add(block);
                    newBlock.add(Blocks.batchFlattenBlock());
                    newBlock.add(Linear.builder().setUnits(NUM_OF_OUTPUT).build());
                    models.put(modelName, model);

                } catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
        return models;
    }

    //Trains MxNetModels so we know which ones can be used for classifications
    public static Map<String, Model> trainMxNetModels(boolean overwrite, int limit) throws ModelNotFoundException, MalformedModelException, IOException {
        List<String> modelNames = new ArrayList<String>();
        Map<String, Model> models = new HashMap<String, Model>();
        List<Artifact> imageClassificationModelList = ModelZoo.listModels().get(CV.IMAGE_CLASSIFICATION);//ModelZoo.listModels().get(CV.IMAGE_CLASSIFICATION);
        int i = 0;
        Set<String> trainedModels = listFilesUsingJavaIO("availableModels");
        String fileEnding = "-0001.params";
        for(Artifact artifact : imageClassificationModelList) {
            if(i > limit && limit != -1){
                continue;
            }
            i++;
            //Map<String, String> filters = new HashMap<>();
            String artifactId = artifact.getMetadata().getArtifactId();
            List<Artifact> subArtifacts = artifact.getMetadata().getArtifacts();
            for(Artifact subArtifact : subArtifacts){
                try{
                    Map<String, String> filters = subArtifact.getProperties();
                    String modelName = subArtifact.getName();
                    for (Map.Entry<String, String> set :
                        filters.entrySet()) {
                        // Printing all elements of a Map
                        modelName = modelName + set.getKey() + "_" + set.getValue() + "_";
                    }
                    if(modelNames.isEmpty() || modelNames.contains(modelName) != true){
                        modelNames.add(modelName);
                    } else{
                        continue;
                    }
                    String modelFile = modelName + fileEnding;
                    if(trainedModels.contains(modelFile) == true && overwrite == false){
                        continue;
                    }
                    
                    System.out.println(modelName);
                    Criteria<Image, Classifications> criteria = Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optArtifactId(artifactId)
                        .optFilters(filters)
                        .build();
                    Model model = criteria.loadModel();
                    SequentialBlock newBlock = new SequentialBlock();
                    SymbolBlock block = (SymbolBlock) model.getBlock();
                    block.removeLastBlock();
                    newBlock.add(block);
                    newBlock.add(Blocks.batchFlattenBlock());
                    newBlock.add(Linear.builder().setUnits(NUM_OF_OUTPUT).build());
                    model.setBlock(newBlock);
                    models.put(modelName, model);
                } catch(Exception e){
                    e.printStackTrace();
                }
            }
        }
        return models;
    }

    //Trains MxNetModels so we know which ones can be used for classifications
    public static Map<String, Model> trainCustomModels() throws ModelNotFoundException, MalformedModelException, IOException {
        Map<String, Model> models = new HashMap<String, Model>();
        int filterSize = 8;
        for(int i = 0; i < Models.CUSTOM_MODELS.length; i++){
            String modelName = Models.CUSTOM_MODELS[i];
            Model model = Model.newInstance(modelName);
            
            SequentialBlock block = new SequentialBlock();
            //add some layers
            block.add(Conv2d.builder().setFilters(filterSize).setKernelShape(new Shape(3, 3)).optStride(new Shape(1, 1)).optPadding(new Shape(1, 1)).build());
            block.add(Activation.reluBlock());
            block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
            filterSize = filterSize * 2;

            //addLayers depending on the customModelSize
            for(int j = 0; j <= i; j++){
                block.add(Conv2d.builder().setFilters(filterSize).setKernelShape(new Shape(3, 3)).optStride(new Shape(1, 1)).optPadding(new Shape(1, 1)).build());
                block.add(Activation.reluBlock());
                block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
                filterSize = filterSize * 2;
                //cannot create an infinite amount of filters
                if(filterSize > 1048576){
                    System.out.println(filterSize);
                    filterSize = 2097152;
                }
            }
            filterSize = 8;

            // Reshape the output to a 1D tensor
            block.add(Blocks.batchFlattenBlock()); // This will flatten the output

            block.add(Dropout.builder().optRate(0.3f).build());

            block.add(Linear.builder().setUnits(512).build());
            block.add(Activation.reluBlock());

            block.add(Linear.builder().setUnits(NUM_OF_OUTPUT).build());

            model.setBlock(block);

            models.put(modelName, model);
        }
        return models;
    }

    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }
}
