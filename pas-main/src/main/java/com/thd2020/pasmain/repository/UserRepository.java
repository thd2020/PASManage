package com.thd2020.pasmain.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.thd2020.pasmain.entity.User;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.Optional;
import java.util.List;
import java.time.LocalDateTime;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    // 基本的CRUD操作由JpaRepository提供

    // 自定义查询方法
    Optional<User> findByUsername(String username);

    List<User> findByStatus(User.Status status);

    List<User> findByRole(User.Role role);

    List<User> findByCreatedAtAfter(LocalDateTime date);

    List<User> findByUsernameAndStatus(String username, User.Status status);

    // 使用JPQL进行自定义查询
    @Query("SELECT u FROM User u WHERE u.email = :email")
    Optional<User> findByEmail(@Param("email") String email);

    @Query("SELECT u FROM User u WHERE u.phone = :phone")
    Optional<User> findByPhone(@Param("phone") String phone);

    List<User> findByHospital_HospitalId(Long hospitalId);
}
