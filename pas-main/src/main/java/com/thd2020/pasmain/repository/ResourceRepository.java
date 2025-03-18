package com.thd2020.pasmain.repository;

import com.thd2020.pasmain.entity.Resource;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ResourceRepository extends JpaRepository<Resource, Long> {
    List<Resource> findByCategory(String category);
    List<Resource> findByResourceType(String resourceType);
    List<Resource> findByCategoryAndResourceType(String category, String resourceType);
    @Query(value = "SELECT r FROM Resource r ORDER BY r.timestamp DESC LIMIT :limit")
    List<Resource> findTopByOrderByTimestampDesc(@Param("limit") int limit);
    Optional<Resource> findBySourceUrlAndIdentifier(String fileUrl, String id);
}
