package com.springboot.springbootlogindemo.controller;

import com.springboot.springbootlogindemo.domain.PyTestResult;
import com.springboot.springbootlogindemo.service.PyTestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/py-test")
public class PyTestController {
    private final PyTestService pyTestService;

    @Autowired
    public PyTestController(PyTestService pyTestService) {
        this.pyTestService = pyTestService;
    }

    @GetMapping("/run-and-retrieve")
    public void runPyScriptAndRetrieveResults() {
        pyTestService.runPyScriptAndSaveToDatabase();
    }

    @GetMapping("/results")
    public List<PyTestResult> getPyTestResults() {
        return pyTestService.getPyTestResults();
    }
}
