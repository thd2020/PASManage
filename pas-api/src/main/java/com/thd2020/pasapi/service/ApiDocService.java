package com.thd2020.pasapi.service;

import com.thd2020.pasapi.entity.ApiDoc;
import com.thd2020.pasapi.repository.ApiDocRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.PathItem;
import io.swagger.v3.oas.models.Operation;
import io.swagger.v3.oas.models.media.MediaType;
import io.swagger.v3.oas.models.parameters.Parameter;
import io.swagger.v3.oas.models.responses.ApiResponse;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class ApiDocService {

    @Autowired
    private ApiDocRepository apiDocRepository;

    public List<ApiDoc> getAllApiDocs() {
        return apiDocRepository.findAll();
    }

    public Optional<ApiDoc> getApiDocById(Long id) {
        return apiDocRepository.findById(id);
    }

    public ApiDoc createApiDoc(ApiDoc apiDoc) {
        return apiDocRepository.save(apiDoc);
    }

    public ApiDoc updateApiDoc(Long id, ApiDoc apiDoc) {
        if (apiDocRepository.existsById(id)) {
            apiDoc.setId(id);
            return apiDocRepository.save(apiDoc);
        }
        return null;
    }

    public void deleteApiDoc(Long id) {
        apiDocRepository.deleteById(id);
    }

    public void initializeApiDocsFromSwagger(OpenAPI openAPI) {
        for (Map.Entry<String, PathItem> entry : openAPI.getPaths().entrySet()) {
            String endpoint = entry.getKey();
            PathItem pathItem = entry.getValue();
            for (Map.Entry<PathItem.HttpMethod, Operation> operationEntry : pathItem.readOperationsMap().entrySet()) {
                PathItem.HttpMethod method = operationEntry.getKey();
                Operation operation = operationEntry.getValue();

                ApiDoc apiDoc = new ApiDoc();
                apiDoc.setTitle(operation.getSummary());
                apiDoc.setDescription(operation.getDescription());
                apiDoc.setEndpoint(endpoint);
                apiDoc.setMethod(method.name());
                apiDoc.setParams(operation.getParameters().stream()
                        .map(Parameter::getName)
                        .collect(Collectors.joining(", ")));
                apiDoc.setTypes(operation.getRequestBody().getContent().values().stream()
                        .map(MediaType::getSchema)
                        .map(schema -> schema.getType())
                        .collect(Collectors.joining(", ")));
                apiDoc.setExpectedReturns(operation.getResponses().values().stream()
                        .map(ApiResponse::getDescription)
                        .collect(Collectors.joining(", ")));
                apiDoc.setExamples(operation.getRequestBody().getContent().values().stream()
                        .map(mediaType -> mediaType.getExamples().values().stream()
                                .map(example -> example.getValue().toString())
                                .collect(Collectors.joining(", ")))
                        .collect(Collectors.joining(", ")));

                apiDocRepository.save(apiDoc);
            }
        }
    }
}
