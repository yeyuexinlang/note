# java 数组

## 定义

**一维数组**

```java
/*
声明：
数组元素类型 数组名[];
数组元素类型[] 数组名;
注：声明时[]中不能添加任何数据

为数组分配内存空间：
数组名 = new 数组类型[数组元素个数];

*/
String[] str1;
str1 = new int[5];
int[] arr = new int[10];
int[] arr2 = new int[]{9, 3, 5};
```

**二维数组**

```java
/*
声明：
数组元素类型 数组名[][];
数组元素类型[][] 数组名;
注：声明时[]中不能添加任何数据

为数组分配内存空间：
数组名 = new 数组类型[数组元素个数][数组元素个数];
*/
int[][] arr = new int[10][10];
int[][] arr2 = new int[][]{{2, 5}, {3, 6}};
```

## 显示数组

```java
import java.util.Arrays;
public class try_ {
    public static void main(String[] args){
        int arr[] = new int[]{3, 4, 5, 2};
        // show
        System.out.println(Arrays.toString(arr));
    }
}
/*
[3, 4, 5, 2]
*/
```

## 元素排序

```java
import java.util.Arrays;
public class try_ {
    public static void main(String[] args){
        int arr[] = new int[]{3, 4, 5, 2};
        System.out.println("排序前：" + Arrays.toString(arr));
        Arrays.sort(arr);
        System.out.println("排序后：" + Arrays.toString(arr));
        /*
        手动反转数组
        reverseArray(arr);
        */
    }
}
/*
排序前：[3, 4, 5, 2]
排序后：[2, 3, 4, 5]
*/
```

# 类



# 容器

## ArrayList 集合

```java
import java.util.ArrayList;
public class try_ {
    public static void main(String[] args){
        // 创建集合
        ArrayList<String> arr = new ArrayList<String>();

        //添加元素
        arr.add("a");
        arr.add("b");
        arr.add("c");
        arr.add("e");
        System.out.println(">>>" + arr.toString());

        System.out.println();

        // 删除指定元素
        System.out.println("remove:\"a\" \n>>>" + arr.remove("a"));
        System.out.println(">>>" + arr.toString());

        System.out.println();

        // 删除指定索引的元素
        System.out.println("remove index: 1 \n>>>" + arr.remove(1));
        System.out.println(">>>" + arr.toString());

        System.out.println();

        // 修改指定索引元素
        System.out.println("修改指定索引处的元素，返回修改的元素: 1,\"k\" \n>>>" + arr.set(1, "k"));
        System.out.println(">>>" + arr.toString());

        System.out.println();

        // 获取指定索引元素
        System.out.println("返回指定索引处的元素：1 \n>>>" + arr.get(1));
        System.out.println(">>>" + arr.toString());

        System.out.println();

        // 获取元素数量
        System.out.println("获取元素数量 \n>>>" + arr.size());
        System.out.println(">>>" + arr.toString());
    }
}

/*
>>>[a, b, c, e]

remove:"a" 
>>>true
>>>[b, c, e]

remove index: 1 
>>>c
>>>[b, e]

修改指定索引处的元素，返回修改的元素: 1,"k" 
>>>e
>>>[b, k]

返回指定索引处的元素：1 
>>>k
>>>[b, k]

获取元素数量 
>>>2
>>>[b, k]
*/



```