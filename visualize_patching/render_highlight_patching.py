
from prettify_js import get_js_library
from PIL import Image
import io
import imgkit
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
def load_code(path):
    code = []
    print(path)
    with open(path, "r") as fin:
        for l in fin:
            code.append(l)
    return "".join(code)


def from_to_file(in_path, out_path, lang):
    code = load_code(in_path)
    image = render(code, lang, out_path)
    #image.save(out_path)


def render(code, lang, out_path):
    # uses https://github.com/google/code-prettify for syntax highlighting
    jslib = get_js_library()
    html_page = """
        <!DOCTYPE html><html>
        <script>"""+ jslib + """</script>
        <!-- custom version of sunburst style, originally by David Leibovic -->
        <style type="text/css">
            pre .str, code .str { color: #65B042; } /* string  - green */
            pre .kwd, code .kwd { color: #E28964; } /* keyword - dark pink */
            pre .com, code .com { color: #AEAEAE; font-style: italic; } /* comment - gray */
            pre .typ, code .typ { color: #89bdff; } /* type - light blue */
            pre .lit, code .lit { color: #3387CC; } /* literal - blue */
            pre .pun, code .pun { color: #000; } /* punctuation - black */
            pre .pln, code .pln { color: #000; } /* plaintext - black */
            pre .tag, code .tag { color: #89bdff; } /* html/xml tag    - light blue */
            pre .atn, code .atn { color: #bdb76b; } /* html/xml attribute name  - khaki */
            pre .atv, code .atv { color: #65B042; } /* html/xml attribute value - green */
            pre .dec, code .dec { color: #3387CC; } /* decimal - blue */
    
            body {
                margin: 0px;
            }
            pre {
                margin: 0px;
            }
            
            pre.prettyprint, code.prettyprint {
                background-color: #fff;
                padding: none;
                border: none;
                margin: none;
            }
    
            /* Specify class=linenums on a pre to get line numbering */
            ol.linenums { margin-top: 0; margin-bottom: 0; color: #AEAEAE; } /* IE indents via margin-left */
            li.L0,li.L1,li.L2,li.L3,li.L5,li.L6,li.L7,li.L8 { list-style-type: none }
            /* Alternate shading for lines */
            li.L1,li.L3,li.L5,li.L7,li.L9 { }
    
            @media print {
              pre .str, code .str { color: #060; }
              pre .kwd, code .kwd { color: #006; font-weight: bold; }
              pre .com, code .com { color: #600; font-style: italic; }
              pre .typ, code .typ { color: #404; font-weight: bold; }
              pre .lit, code .lit { color: #044; }
              pre .pun, code .pun { color: #440; }
              pre .pln, code .pln { color: #000; }
              pre .tag, code .tag { color: #006; font-weight: bold; }
              pre .atn, code .atn { color: #404; }
              pre .atv, code .atv { color: #060; }
            }
        </style>
        <?prettify lang=%s linenums=false?>
        <pre class="prettyprint lang-java mycode">""" % lang + code.replace("\t", "    ") + """    </pre>
    </html>"""
    options = {
        'format': 'png',
        'quiet': '',
	'width': 50,
    }
    with open("generate.html", "w") as fp:
        fp.write(html_page)
    #print(html_page)
    from html2image import Html2Image
    hti = Html2Image()
    css = "body {background: white;}"
    hti.screenshot(html_str=html_page, css_str=css, save_as=out_path)
    #image_raw = imgkit.from_string(html_page, False, options)
    #return Image.open(io.BytesIO(image_raw))

def divide(img,m,n):
    h, w = img.shape[0],img.shape[1]
    grid_h=int(h*1.0/(m-1)+0.5)
    grid_w=int(w*1.0/(n-1)+0.5)
    h=grid_h*(m-1)
    w=grid_w*(n-1)
    img_re=cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m))
    gx=gx.astype(np.int)
    gy=gy.astype(np.int)
    divide_image = np.zeros([m-1, n-1, grid_h, grid_w,3], np.uint8)
    for i in range(m-1):
        for j in range(n-1):
            divide_image[i,j,...]=img_re[
            gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]
    return divide_image

def display_blocks(divide_image):#
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j,:])
            img = Image.fromarray(divide_image[i, j, :], "RGB")
            img.save(str(i)+'.jpg')
            plt.axis('off')
    plt.show()
def render_one_file():
    from_to_file("test.java", "generate.png", "Java") # Replace with the path to the source code file and path to store the png
    with open("test.java", "r") as fp:  # Replace with the input source code file
        lines = fp.readlines()
        len = len(''.join(lines).split('\n'))
    img = cv2.imread('generate.png')
    img = img[0:0+int(len*15),0:1000]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[0], img.shape[1]
    fig1 = plt.figure('Image')
    cv2.imshow(winname="img", mat=img)
    plt.axis('off')
    plt.title('Original image')
    print('\t\t\t   shape:\n', '\t\t\t', img.shape)
    divide_img = divide(img, len + 1, 2)
    display_blocks(divide_img)

def render_all_files(path, out_path):
    with open(path, 'r', encoding='utf-8') as fp:
        lines = fp.read().splitlines()
        for i in range(0, len(lines)):
            if not os.path.exists('renderings/' + str(i)):
               os.mkdir('renderings/' + str(i))
            if i > 10:
                break
            cur = json.loads(lines[i])
            code = cur['code']
            image = render(code, "Java", out_path)
            code_len = len(code.split('\n'))
            img = cv2.imread('generate.png')
            img = img[0:0 + int(code_len * 15), 0:1000]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            full_img = Image.fromarray(img, "RGB")
            full_img.save(r'renderings\\' + str(i) + r'\\' + str(i) + '_' + str(0) + '.jpg')
            divide_image = divide(img, code_len + 1, 2)
            m, n = divide_image.shape[0], divide_image.shape[1]
            for k in range(m):
                for j in range(n):
                    img = Image.fromarray(divide_image[k, j, :], "RGB")
                    img.save(r'renderings\\'+str(i)+r'\\'+str(i)+'_'+str(k+1) + '.jpg')

if __name__ == "__main__":
    render_all_files('train.jsonl', "generate.png")



