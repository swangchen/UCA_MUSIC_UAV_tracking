package com.springboot.springbootlogindemo.domain;
import javax.persistence.*;

@Table(name = "uavimage")
@Entity
public class UAVImage {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "name")
    private Integer name;

    @Column(name = "class1")
    private Integer class1;

    @Column(name = "class2")
    private Integer class2;

    @Column(name = "weight")
    private Float weight;

    @Column(name = "path", length = 255)
    private String path;

    // 以下是构造函数、getter和setter方法，以及其他需要的方法

    public UAVImage() {
        // 默认构造函数
    }

    // 其他构造函数

    // Getter和Setter方法

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Integer getName() {
        return name;
    }

    public void setName(Integer name) {
        this.name = name;
    }

    public Integer getClass1() {
        return class1;
    }

    public void setClass1(Integer class1) {
        this.class1 = class1;
    }

    public Integer getClass2() {
        return class2;
    }

    public void setClass2(Integer class2) {
        this.class2 = class2;
    }

    public Float getWeight() {
        return weight;
    }

    public void setWeight(Float weight) {
        this.weight = weight;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }
}
