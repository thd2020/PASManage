package com.thd2020.pasmain.util;

import com.thd2020.pasmain.entity.User;
import com.thd2020.pasmain.service.UserService;
import org.springframework.beans.BeanWrapper;
import org.springframework.beans.BeanWrapperImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

@Service
public class UtilFunctions {

    @Autowired
    private JwtUtil jwtUtil;
    @Autowired
    private UserService userService;

    // 获取对象中非空字段的名称
    public String[] getNullPropertyNames(Object source) {
        final BeanWrapper src = new BeanWrapperImpl(source);
        java.beans.PropertyDescriptor[] pds = src.getPropertyDescriptors();

        Set<String> emptyNames = new HashSet<>();
        for (java.beans.PropertyDescriptor pd : pds) {
            Object srcValue = src.getPropertyValue(pd.getName());
            if (srcValue == null) emptyNames.add(pd.getName());
        }
        String[] result = new String[emptyNames.size()];
        return emptyNames.toArray(result);
    }

    public Boolean isAdmin(String token){
        Long userId = jwtUtil.extractUserId(token.substring(7));
        Optional<User> requestingUser = userService.getUserById(userId);
        return requestingUser.isPresent() && requestingUser.get().getRole() == User.Role.ADMIN;
    }

    public Boolean isDoctor(String token){
        Long userId = jwtUtil.extractUserId(token.substring(7));
        Optional<User> requestingUser = userService.getUserById(userId);
        return requestingUser.isPresent() && (requestingUser.get().getRole() == User.Role.B_DOCTOR || requestingUser.get().getRole() == User.Role.T_DOCTOR);
    }

    public Boolean isMatch(String token, Long id){
        Long userId = jwtUtil.extractUserId(token.substring(7));
        Optional<User> requestingUser = userService.getUserById(userId);
        return requestingUser.isPresent() && Objects.equals(requestingUser.get().getUserId(), id);
    }
}
