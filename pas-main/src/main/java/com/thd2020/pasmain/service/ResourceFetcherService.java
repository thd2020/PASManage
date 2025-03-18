package com.thd2020.pasmain.service;

import com.thd2020.pasmain.repository.ResourceRepository;
import com.thd2020.pasmain.entity.Resource;
import com.thd2020.pasmain.exception.ResourceNotFoundException;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
public class ResourceFetcherService {
    
    @Value("${resource.fetch.enabled:true}")
    private boolean fetchEnabled;

    @Value("${resource.download.path}")
    private String downloadPath;

    @Value("${pubmed.api.key:}")
    private String pubmedApiKey;

    @Value("${baidu.api.key:}")
    private String baiduApiKey;

    @Value("${bing.api.key:}")
    private String bingApiKey;

    @Value("${bilibili.cookie:}")
    private String bilibiliCookie;

    @Value("${baidu.search.url}")
    private String baiduSearchUrl;

    @Value("${bing.search.url}")
    private String bingSearchUrl;

    @Value("${bilibili.search.url}")
    private String bilibiliSearchUrl;

    private final ResourceRepository resourceRepository;
    private final List<String> sourceUrls;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
            
                public ResourceFetcherService(ResourceRepository resourceRepository) {
                    this.resourceRepository = resourceRepository;
                    this.restTemplate = new RestTemplate();
                    this.objectMapper = new ObjectMapper();
                    // Configure source URLs
                    this.sourceUrls = List.of(
                        "https://api.pubmed.ncbi.nlm.nih.gov/search/placenta_accreta",
                        "https://api.clinicaltrials.gov/api/query/study_fields?expr=placenta+accreta",
                        "baidu:placenta_accreta",
                        "bing:placenta_accreta",
                        "bilibili:placenta_accreta"
                    );
                }
            
                @Scheduled(fixedRateString = "${resource.fetch.interval:3600000}") // Default 1 hour
                public void fetchResources() {
                    if (!fetchEnabled) return;
                    
                    for (String sourceUrl : sourceUrls) {
                        try {
                            List<Resource> newResources = fetchResourcesFromUrl(sourceUrl);
                            for (Resource resource : newResources) {
                                if (!resourceExists(resource)) {
                                    resourceRepository.save(resource);
                                }
                            }
                        } catch (IOException e) {
                            // Log error but continue with next URL
                            e.printStackTrace();
                        }
                    }
                }
            
                private boolean resourceExists(Resource resource) {
                    return resourceRepository.findBySourceUrlAndIdentifier(
                        resource.getSourceUrl(),
                        resource.getIdentifier()
                    ).isPresent();
                }
            
                private List<Resource> fetchResourcesFromUrl(String sourceUrl) throws IOException {
                    // Implement different fetching strategies based on URL
                    if (sourceUrl.contains("pubmed")) {
                        return fetchFromPubMed(sourceUrl);
                    } else if (sourceUrl.contains("clinicaltrials")) {
                        return fetchFromClinicalTrials(sourceUrl);
                    } else if (sourceUrl.startsWith("baidu:")) {
                        return fetchFromBaidu(sourceUrl.substring(6));
                    } else if (sourceUrl.startsWith("bing:")) {
                        return fetchFromBing(sourceUrl.substring(5));
                    } else if (sourceUrl.startsWith("bilibili:")) {
                        return fetchFromBilibili(sourceUrl.substring(9));
                    }
                    return List.of();
                }
            
                private List<Resource> fetchFromPubMed(String url) throws IOException {
                    HttpHeaders headers = new HttpHeaders();
                    if (!pubmedApiKey.isEmpty()) {
                        headers.set("api-key", pubmedApiKey);
                    }
                    
                    ResponseEntity<String> response = restTemplate.exchange(
                        url,
                        HttpMethod.GET,
                        new HttpEntity<>(headers),
                        String.class
                    );
            
                    JsonNode root = objectMapper.readTree(response.getBody());
                    List<Resource> resources = new ArrayList<>();
                    
                    JsonNode results = root.path("esearchresult").path("idlist");
                    for (JsonNode id : results) {
                        String articleUrl = "https://pubmed.ncbi.nlm.nih.gov/" + id.asText();
                        String detailsUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=" 
                                           + id.asText() + "&retmode=json";
                        
                        ResponseEntity<String> details = restTemplate.getForEntity(detailsUrl, String.class);
                        JsonNode articleData = objectMapper.readTree(details.getBody())
                                                        .path("result")
                                                        .path(id.asText());
            
                        Resource resource = new Resource();
                        resource.setSourceUrl(articleUrl);
                        resource.setIdentifier(id.asText());
                        resource.setTitle(articleData.path("title").asText());
                        resource.setResourceType("PUBLICATION");
                        resource.setCategory("RESEARCH");
                        resource.setTimestamp(LocalDateTime.now());
                        
                        resources.add(resource);
                    }
                    
                    return resources;
                }
            
                private List<Resource> fetchFromClinicalTrials(String url) throws IOException {
                    ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
                    JsonNode root = objectMapper.readTree(response.getBody());
                    List<Resource> resources = new ArrayList<>();
                    
                    JsonNode studies = root.path("StudyFieldsResponse").path("StudyFields");
                    for (JsonNode study : studies) {
                        Resource resource = new Resource();
                        resource.setIdentifier(study.path("NCTId").get(0).asText());
                        resource.setSourceUrl("https://clinicaltrials.gov/study/" + resource.getIdentifier());
                        resource.setTitle(study.path("BriefTitle").get(0).asText());
                        resource.setResourceType("CLINICAL_TRIAL");
                        resource.setCategory("RESEARCH");
                        resource.setTimestamp(LocalDateTime.now());
                        
                        resources.add(resource);
                    }
                    
                    return resources;
                }
            
                private List<Resource> fetchFromBaidu(String query) throws IOException {
                    if (baiduApiKey.isEmpty()) return List.of();
            
                    HttpHeaders headers = new HttpHeaders();
                    headers.set("apikey", baiduApiKey);
                    
                    String url = String.format("%s?q=%s", baiduSearchUrl, query);
                ResponseEntity<String> response = restTemplate.exchange(
                    url,
                    HttpMethod.GET,
                    new HttpEntity<>(headers),
                    String.class
                );
        
                JsonNode root = objectMapper.readTree(response.getBody());
                List<Resource> resources = new ArrayList<>();
                
                for (JsonNode result : root.path("items")) {
                    Resource resource = new Resource();
                    resource.setSourceUrl(result.path("link").asText());
                    resource.setIdentifier("baidu_" + result.path("id").asText());
                    resource.setTitle(result.path("title").asText());
                    resource.setResourceType("ARTICLE");
                    resource.setCategory("SEARCH_RESULT");
                    resource.setTimestamp(LocalDateTime.now());
                    resources.add(resource);
                }
                
                return resources;
            }
        
            private List<Resource> fetchFromBing(String query) throws IOException {
                if (bingApiKey.isEmpty()) return List.of();
        
                HttpHeaders headers = new HttpHeaders();
                headers.set("Ocp-Apim-Subscription-Key", bingApiKey);
                
                String url = String.format("%s?q=%s", bingSearchUrl, query);
        ResponseEntity<String> response = restTemplate.exchange(
            url,
            HttpMethod.GET,
            new HttpEntity<>(headers),
            String.class
        );

        JsonNode root = objectMapper.readTree(response.getBody());
        List<Resource> resources = new ArrayList<>();
        
        for (JsonNode result : root.path("webPages").path("value")) {
            Resource resource = new Resource();
            resource.setSourceUrl(result.path("url").asText());
            resource.setIdentifier("bing_" + result.path("id").asText());
            resource.setTitle(result.path("name").asText());
            resource.setResourceType("WEBPAGE");
            resource.setCategory("SEARCH_RESULT");
            resource.setTimestamp(LocalDateTime.now());
            resources.add(resource);
        }
        
        return resources;
    }

    private List<Resource> fetchFromBilibili(String query) throws IOException {
        HttpHeaders headers = new HttpHeaders();
        if (!bilibiliCookie.isEmpty()) {
            headers.set("Cookie", bilibiliCookie);
        }
        
        String url = String.format("%s?keyword=%s&search_type=video", bilibiliSearchUrl, query);
        ResponseEntity<String> response = restTemplate.exchange(
            url,
            HttpMethod.GET,
            new HttpEntity<>(headers),
            String.class
        );

        JsonNode root = objectMapper.readTree(response.getBody());
        List<Resource> resources = new ArrayList<>();
        
        for (JsonNode result : root.path("data").path("result")) {
            Resource resource = new Resource();
            resource.setSourceUrl("https://www.bilibili.com/video/" + result.path("bvid").asText());
            resource.setIdentifier("bilibili_" + result.path("aid").asText());
            resource.setTitle(result.path("title").asText());
            resource.setResourceType("VIDEO");
            resource.setCategory("EDUCATIONAL");
            resource.setTimestamp(LocalDateTime.now());
            resources.add(resource);
        }
        
        return resources;
    }

    public String downloadResource(Long resourceId) throws IOException {
        Resource resource = resourceRepository.findById(resourceId)
            .orElseThrow(() -> new ResourceNotFoundException("Resource not found"));

        String fileName = resource.getIdentifier() + "_" + 
                         resource.getTitle().replaceAll("[^a-zA-Z0-9]", "_") + ".pdf";
        Path targetPath = Paths.get(downloadPath, fileName);
        
        if (!Files.exists(targetPath)) {
            Files.createDirectories(targetPath.getParent());
            
            byte[] content = restTemplate.getForObject(resource.getSourceUrl(), byte[].class);
            Files.write(targetPath, content);
            
            resource.setLocalPath(targetPath.toString());
            resourceRepository.save(resource);
        }
        
        return targetPath.toString();
    }
}
