# JPEG decoder made in Python

By: Tiago Becerra Paolini

## About

This decoder supports both baseline (sequential) and progressive JPEG images. The decoder was made for the purpose of learning how JPEG images work, and I am making it available so it may help others to also learn about. The primary goal of this project is code readability, not performance (furthermore, one would not make a media decoder purely in an interpreted language). But some optimizations were made so it takes a reasonable amount of time to run.

My motivation was that it was quite difficult to find online detailed explanations on how to actually implement JPEG in code, most of what I found were just general explanations of the concepts involved. Specially when it comes to progressive JPEG, most of what I found about were a description of ***what*** it accomplishes but not ***how*** it accomplishes. But I am glad that after a great deal of research and effort I managed to pull off the decoding of progressive JPEG images, not only the usual sequential images.

I thoroughly commented my code to explain what is being accomplished is each step, so I encourage you to go through the source code, as Python is not difficult to read. I intend at some point to create a tutorial. But for the time being, I would like to point to the references that I used for the project, which explain in even greater detail how JPEG decoding works.

## References

* [Compressed Image File Formats](https://research-solution.com/uplode/book/book-26184.pdf), book by John Miano: it has the most comprehensive explanation of JPEG decoding that I found, if you only have time to look at one reference in this list then please have a look at it. The book works its way through from the fundamentals until the finer details of the decoding process, all in a didactic manner. It was also the only place I found an explanation on how to decode progressive JPEG (if you do not count the [original JPEG specification](https://www.w3.org/Graphics/JPEG/itu-t81.pdf), but that is a much harder read).
* [JPEG Huffman Coding Tutorial](https://www.impulseadventure.com/photo/jpeg-huffman-coding.html), by Calvin Hass: an excellent explanation on how Huffman trees are used in the context of JPEG, I recommend the read in order to understand what Huffman coding is and how it works. Unfortunately, the tutorial is lacking when it comes to the decoding of AC coefficients, but that is covered in the book I recommended above. The tutorial still is a very helpful read, nonetheless.
* [Understanding and Decoding a JPEG Image using Python](https://yasoob.me/posts/understanding-and-writing-jpeg-decoder-in-python/), by Yasoob Khalid: a great reference to understand and visualize what is happening through the decoding process. The page has plenty of illustrations, of particular note the layout of a JPEG file. It goes in length to explain the ideas behind the compression techniques used in JPEG. However, the text abruptly jumps to the implementation without too much explanation on how those ideas are applied in the context of JPEG, it is mostly just the code with little commentary. Even so, it is a good read for getting started with JPEG decoding.
* [Digital Compression and Coding of Continuous-Tone Still Images - Requirements and Guidelines](https://www.w3.org/Graphics/JPEG/itu-t81.pdf): this is the original official JPEG specification (back from 1992). It does not define a file format or color spaces, it is just the encoding techniques for JPEG images. Many of the techniques recommended there never really got widespread usage. The reason is a either those techniques required (at the time) the licensing of patents, and/or that the technique would have poor performance in the hardware of the time. Pretty much the only techniques that actually matter today are Baseline DCT and Progressive DCT, both with Huffman coding. Even though by now the patents have expired and the hardware greatly improved, the remaining techniques are unlikely to find general use due to the lack of support and that newer technologies already accomplish what they set out to do. Also Baseline and Progressive JPEG already do a great job, and this specification is a the canon reference to them.
* [JPEG File Interchange Format](https://www.w3.org/Graphics/JPEG/jfif3.pdf): it is the original official specification of a file format and a color space for JPEG images, both of which was lacking in the above specification.

## Usage

The decoder requires the following third party Python modules:
* [Numpy](https://numpy.org/install/)
* [Scipy](https://scipy.org/install/)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html) *(only for displaying the decoded image, the actual decoding process is still made by my decoder)*

You can install all of them at once with the following shell command (if you already have Python installed):
```shell
pip3 install numpy scipy pillow
```

You can [download Python here](https://www.python.org/downloads/). The decoder was made in Python 3.9.4, but it works in later versions (and it is assumed to also work on previous 3.x versions).