import org.python.core.PyFunction;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;
import java.io.BufferedReader;  
import java.io.File;  
import java.io.InputStreamReader; 

public class HelloPython {
    public static void main(String[] args) {
        try{
        //PythonInterpreter interpreter = new PythonInterpreter();
        //interpreter.execfile("./demo.py"); 

        /* for(int i=0;i<10;i++){
        PyFunction pyFunction = interpreter.get("getCode", PyFunction.class); // 第一个参数为期望获得的函数（变量）的名字，第二个参数为期望返回的对象类型
        PyObject pyObject = pyFunction.__call__(); // 调用函数

        System.out.println(pyObject.toString());
        } */
            String[] argg = new String[] { "python", "/data/sata/share_sata/xyq/crnn/recognition.py"};  
              Process pr = Runtime.getRuntime().exec(argg);  
      
              BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));  
              String line;  
              String result = "";
              String result2 = "";              
              while ((line = in.readLine()) != null) {  
    //                line = decodeUnicode(line);  
                result += line;
                result2 = line;
              }  
              in.close();  
              pr.waitFor();  
              System.out.println("识别结果："+result2);
            }
        catch(Exception e){
            System.out.println(e);
        }
        
    }
}