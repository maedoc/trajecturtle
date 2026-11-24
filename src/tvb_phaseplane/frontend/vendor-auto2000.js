// ═══════════════════════════════════════════════════════════════
//  AUTO2000 WASM Continuation Core (Emscripten + JS wrapper)
//  Vendored from PyDSTool PyCont/auto (BSD-3-Clause, zero GPL)
// ═══════════════════════════════════════════════════════════════

const createAuto2000Core = (() => {
  
  return (
async function(moduleArg = {}) {
  var moduleRtn;

var Module=moduleArg;var readyPromiseResolve,readyPromiseReject;var readyPromise=new Promise((resolve,reject)=>{readyPromiseResolve=resolve;readyPromiseReject=reject});var ENVIRONMENT_IS_WEB=typeof window=="object";var ENVIRONMENT_IS_WORKER=typeof WorkerGlobalScope!="undefined";var ENVIRONMENT_IS_NODE=typeof process=="object"&&process.versions?.node&&process.type!="renderer";if(ENVIRONMENT_IS_NODE){const{createRequire}=await import("module");var require=createRequire((typeof self !== "undefined" && self.location && self.location.href ? self.location.href : "file:///"))}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=(typeof self !== "undefined" && self.location && self.location.href ? self.location.href : "file:///");var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");var nodePath=require("path");if(_scriptName.startsWith("file:")){scriptDirectory=nodePath.dirname(require("url").fileURLToPath(_scriptName))+"/"}readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\/g,"/")}arguments_=process.argv.slice(2);quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}{if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var out=console.log.bind(console);var err=console.error.bind(console);var wasmBinary;var wasmMemory;var ABORT=false;var EXITSTATUS;var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAP64,HEAPU64,HEAPF64;var runtimeInitialized=false;var isFileURI=filename=>filename.startsWith("file://");function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);Module["HEAP32"]=HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);Module["HEAPF64"]=HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function postRun(){if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}var runDependencies=0;var dependenciesFulfilled=null;function getUniqueRunDependency(id){return id}function addRunDependency(id){runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)}function removeRunDependency(id){runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);readyPromiseReject(e);throw e}var wasmBinaryFile;function findWasmBinary(){if(Module["locateFile"]){return locateFile("auto2000_core.wasm")}return new URL("auto2000_core.wasm",(typeof self !== "undefined" && self.location && self.location.href ? self.location.href : "file:///")).href}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&typeof WebAssembly.instantiateStreaming=="function"&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){return{env:wasmImports,wasi_snapshot_preview1:wasmImports}}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmMemory=wasmExports["memory"];updateMemoryViews();removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(mod,inst)=>{resolve(receiveInstance(mod,inst))})})}wasmBinaryFile??=findWasmBinary();try{var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}catch(e){readyPromiseReject(e);return Promise.reject(e)}}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var noExitRuntime=true;var UTF8Decoder=typeof TextDecoder!="undefined"?new TextDecoder:undefined;var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead=NaN)=>{var endIdx=idx+maxBytesToRead;var endPtr=idx;while(heapOrArray[endPtr]&&!(endPtr>=endIdx))++endPtr;if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.subarray(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var UTF8ToString=(ptr,maxBytesToRead)=>ptr?UTF8ArrayToString(HEAPU8,ptr,maxBytesToRead):"";var ___assert_fail=(condition,filename,line,func)=>abort(`Assertion failed: ${UTF8ToString(condition)}, at: `+[filename?UTF8ToString(filename):"unknown filename",line,func?UTF8ToString(func):"unknown function"]);var syscallGetVarargI=()=>{var ret=HEAP32[+SYSCALLS.varargs>>2];SYSCALLS.varargs+=4;return ret};var syscallGetVarargP=syscallGetVarargI;var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\/]+|\/)\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>{if(ENVIRONMENT_IS_NODE){var nodeCrypto=require("crypto");return view=>nodeCrypto.randomFillSync(view)}return view=>crypto.getRandomValues(view)};var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.charCodeAt(i);if(u>=55296&&u<=57343){var u1=str.charCodeAt(++i);u=65536+((u&1023)<<10)|u1&1023}if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(typeof window!="undefined"&&typeof window.prompt=="function"){result=window.prompt("Input: ");if(result!==null){result+="\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var mmapAlloc=size=>{abort()};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===HEAP8.buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===HEAP8.buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}HEAP8.set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var preloadPlugins=[];var FS_handledByPreloadPlugin=(byteArray,fullname,finish,onerror)=>{if(typeof Browser!="undefined")Browser.init();var handled=false;preloadPlugins.forEach(plugin=>{if(handled)return;if(plugin["canHandle"](fullname)){plugin["handle"](byteArray,fullname,finish,onerror);handled=true}});return handled};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);function processData(byteArray){function finish(byteArray){preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}onload?.();removeRunDependency(dep)}if(FS_handledByPreloadPlugin(byteArray,fullname,finish,()=>{onerror?.();removeRunDependency(dep)})){return}finish(byteArray)}addRunDependency(dep);if(typeof url=="string"){asyncLoad(url).then(processData,onerror)}else{processData(url)}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}mounts.forEach(mount=>{if(!mount.type.syncfs){return done(null)}mount.type.syncfs(mount,populate,done)})},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);Object.keys(FS.nameTable).forEach(hash=>{var current=FS.nameTable[hash];while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}});node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){throw new Error(`Invalid encoding type "${opts.encoding}"`)}var ret;var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){ret=UTF8ArrayToString(buf)}else if(opts.encoding==="binary"){ret=buf}FS.close(stream);return ret},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){var buf=new Uint8Array(lengthBytesUTF8(data)+1);var actualNumBytes=stringToUTF8Array(data,buf,0,buf.length);FS.write(stream,buf,0,actualNumBytes,undefined,opts.canOwn)}else if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{throw new Error("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(typeof XMLHttpRequest!="undefined"){throw new Error("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url);obj.usedBytes=obj.contents.length}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))throw new Error("Couldn't load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)throw new Error("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)throw new Error("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))throw new Error("Couldn't load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")throw new Error("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(typeof XMLHttpRequest!="undefined"){if(!ENVIRONMENT_IS_WORKER)throw"Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc";var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};var keys=Object.keys(node.stream_ops);keys.forEach(key=>{var fn=node.stream_ops[key];stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}});function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,HEAP8,ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){HEAP32[buf>>2]=stat.dev;HEAP32[buf+4>>2]=stat.mode;HEAPU32[buf+8>>2]=stat.nlink;HEAP32[buf+12>>2]=stat.uid;HEAP32[buf+16>>2]=stat.gid;HEAP32[buf+20>>2]=stat.rdev;HEAP64[buf+24>>3]=BigInt(stat.size);HEAP32[buf+32>>2]=4096;HEAP32[buf+36>>2]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();HEAP64[buf+40>>3]=BigInt(Math.floor(atime/1e3));HEAPU32[buf+48>>2]=atime%1e3*1e3*1e3;HEAP64[buf+56>>3]=BigInt(Math.floor(mtime/1e3));HEAPU32[buf+64>>2]=mtime%1e3*1e3*1e3;HEAP64[buf+72>>3]=BigInt(Math.floor(ctime/1e3));HEAPU32[buf+80>>2]=ctime%1e3*1e3*1e3;HEAP64[buf+88>>3]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){HEAP32[buf+4>>2]=stats.bsize;HEAP32[buf+40>>2]=stats.bsize;HEAP32[buf+8>>2]=stats.blocks;HEAP32[buf+12>>2]=stats.bfree;HEAP32[buf+16>>2]=stats.bavail;HEAP32[buf+20>>2]=stats.files;HEAP32[buf+24>>2]=stats.ffree;HEAP32[buf+28>>2]=stats.fsid;HEAP32[buf+44>>2]=stats.flags;HEAP32[buf+36>>2]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=HEAPU8.slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 12:{var arg=syscallGetVarargP();var offset=0;HEAP16[arg+offset>>1]=2;return 0}case 13:case 14:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();HEAP32[argp>>2]=termios.c_iflag||0;HEAP32[argp+4>>2]=termios.c_oflag||0;HEAP32[argp+8>>2]=termios.c_cflag||0;HEAP32[argp+12>>2]=termios.c_lflag||0;for(var i=0;i<32;i++){HEAP8[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=HEAP32[argp>>2];var c_oflag=HEAP32[argp+4>>2];var c_cflag=HEAP32[argp+8>>2];var c_lflag=HEAP32[argp+12>>2];var c_cc=[];for(var i=0;i<32;i++){c_cc.push(HEAP8[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();HEAP32[argp>>2]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();HEAP16[argp>>1]=winsize[0];HEAP16[argp+2>>1]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var getHeapMax=()=>2147483648;var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var growMemory=size=>{var b=wasmMemory.buffer;var pages=(size-b.byteLength+65535)/65536|0;try{wasmMemory.grow(pages);updateMemoryViews();return 1}catch(e){}};var _emscripten_resize_heap=requestedSize=>{var oldSize=HEAPU8.length;requestedSize>>>=0;var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var _proc_exit=code=>{EXITSTATUS=code;if(!keepRuntimeAlive()){Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))};var exitJS=(status,implicit)=>{EXITSTATUS=status;_proc_exit(status)};var _exit=exitJS;function _fd_close(fd){try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=HEAPU32[iov>>2];var len=HEAPU32[iov+4>>2];iov+=8;var curr=FS.read(stream,HEAP8,ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);HEAPU32[pnum>>2]=num;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function _fd_seek(fd,offset,whence,newOffset){offset=bigintToI53Checked(offset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);HEAP64[newOffset>>3]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=HEAPU32[iov>>2];var len=HEAPU32[iov+4>>2];iov+=8;var curr=FS.write(stream,HEAP8,ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);HEAPU32[pnum>>2]=num;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}FS.createPreloadedFile=FS_createPreloadedFile;FS.staticInit();MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>";{if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"]}Module["FS"]=FS;function js2000_func_cb(ndim,u,icp,par,ijac,f,dfdu,dfdp){const cb=Module._autoCallbacks&&Module._autoCallbacks.func;if(cb)cb(ndim,u,icp,par,ijac,f,dfdu,dfdp)}function js2000_stpnt_cb(ndim,t,u,par){const cb=Module._autoCallbacks&&Module._autoCallbacks.stpnt;if(cb)cb(ndim,t,u,par)}var wasmImports={__assert_fail:___assert_fail,__syscall_fcntl64:___syscall_fcntl64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,emscripten_resize_heap:_emscripten_resize_heap,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,js2000_func_cb,js2000_stpnt_cb};var wasmExports=await createWasm();var ___wasm_call_ctors=wasmExports["__wasm_call_ctors"];var _auto2000_init=Module["_auto2000_init"]=wasmExports["auto2000_init"];var _malloc=Module["_malloc"]=wasmExports["malloc"];var _free=Module["_free"]=wasmExports["free"];var _auto2000_set_restart_sp=Module["_auto2000_set_restart_sp"]=wasmExports["auto2000_set_restart_sp"];var _auto2000_set_restart_bvp_sp=Module["_auto2000_set_restart_bvp_sp"]=wasmExports["auto2000_set_restart_bvp_sp"];var _auto2000_run=Module["_auto2000_run"]=wasmExports["auto2000_run"];var _auto2000_num_u=Module["_auto2000_num_u"]=wasmExports["auto2000_num_u"];var _auto2000_u=Module["_auto2000_u"]=wasmExports["auto2000_u"];var _auto2000_usm=Module["_auto2000_usm"]=wasmExports["auto2000_usm"];var _auto2000_par=Module["_auto2000_par"]=wasmExports["auto2000_par"];var _auto2000_ev_r=Module["_auto2000_ev_r"]=wasmExports["auto2000_ev_r"];var _auto2000_ev_i=Module["_auto2000_ev_i"]=wasmExports["auto2000_ev_i"];var _auto2000_num_sp=Module["_auto2000_num_sp"]=wasmExports["auto2000_num_sp"];var _auto2000_sp_itp=Module["_auto2000_sp_itp"]=wasmExports["auto2000_sp_itp"];var _auto2000_sp_lab=Module["_auto2000_sp_lab"]=wasmExports["auto2000_sp_lab"];var _auto2000_sp_nar=Module["_auto2000_sp_nar"]=wasmExports["auto2000_sp_nar"];var _auto2000_sp_nparx=Module["_auto2000_sp_nparx"]=wasmExports["auto2000_sp_nparx"];var _auto2000_sp_ntst=Module["_auto2000_sp_ntst"]=wasmExports["auto2000_sp_ntst"];var _auto2000_sp_ncol=Module["_auto2000_sp_ncol"]=wasmExports["auto2000_sp_ncol"];var _auto2000_sp_ntpl=Module["_auto2000_sp_ntpl"]=wasmExports["auto2000_sp_ntpl"];var _auto2000_sp_nfpr=Module["_auto2000_sp_nfpr"]=wasmExports["auto2000_sp_nfpr"];var _auto2000_sp_has_mesh=Module["_auto2000_sp_has_mesh"]=wasmExports["auto2000_sp_has_mesh"];var _auto2000_sp_icp=Module["_auto2000_sp_icp"]=wasmExports["auto2000_sp_icp"];var _auto2000_sp_rldot=Module["_auto2000_sp_rldot"]=wasmExports["auto2000_sp_rldot"];var _auto2000_sp_par=Module["_auto2000_sp_par"]=wasmExports["auto2000_sp_par"];var _auto2000_sp_u=Module["_auto2000_sp_u"]=wasmExports["auto2000_sp_u"];var _auto2000_sp_t=Module["_auto2000_sp_t"]=wasmExports["auto2000_sp_t"];var _auto2000_sp_ups=Module["_auto2000_sp_ups"]=wasmExports["auto2000_sp_ups"];var _auto2000_sp_udotps=Module["_auto2000_sp_udotps"]=wasmExports["auto2000_sp_udotps"];var ___trap=wasmExports["__trap"];var __emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];function run(){if(runDependencies>0){dependenciesFulfilled=run;return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();readyPromiseResolve(Module);Module["onRuntimeInitialized"]?.();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}function preInit(){if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}preInit();run();moduleRtn=readyPromise;


  return moduleRtn;
}
);
})();

function itpToType(itp) {
  if (itp === 1) return 'BP';
  if (itp === 2) return 'LP';
  if (itp === 3) return 'HB';
  if (itp === 4) return 'RG';
  if (itp === 5) return 'LPC';
  if (itp === 6) return 'BPC';
  if (itp === 7) return 'PD';
  if (itp === 8) return 'TR';
  if (itp === 9) return 'EP';
  if (itp === -32) return 'GH';
  if (itp < 0) {
    const base = Math.abs(itp) % 10;
    if (base === 4) return 'UZ';
    if (base === 9) return 'MX';
    return String(itp);
  }
  if (itp >= 10) {
    const base = itp % 10;
    if (base === 1) return 'BP';
    if (base === 2) return 'LP';
    if (base === 3) return 'HB';
    if (base === 4) return 'RG';
    if (base === 5) return 'LPC';
    if (base === 6) return 'BPC';
    if (base === 7) return 'PD';
    if (base === 8) return 'TR';
    if (base === 9) return 'EP';
  }
  return String(itp);
}

let cachedModulePromise = null;

async function compileWasm(wasmSource = null) {
  if (wasmSource instanceof WebAssembly.Module) {
    return wasmSource;
  }
  if (wasmSource instanceof Uint8Array || wasmSource instanceof ArrayBuffer) {
    return WebAssembly.compile(wasmSource);
  }
  const url = wasmSource ?? new URL('./auto2000_core.wasm', (typeof self !== "undefined" && self.location && self.location.href ? self.location.href : "file:///"));
  if (typeof process !== 'undefined' && process.versions?.node && url instanceof URL && url.protocol === 'file:') {
    const fs = await import('node:fs/promises');
    return WebAssembly.compile(await fs.readFile(url));
  }
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch auto2000 WASM (${resp.status} ${resp.statusText})`);
  }
  if (typeof WebAssembly.compileStreaming === 'function') {
    try {
      return await WebAssembly.compileStreaming(resp);
    } catch {
      const resp2 = await fetch(url);
      return WebAssembly.compile(await resp2.arrayBuffer());
    }
  }
  return WebAssembly.compile(await resp.arrayBuffer());
}

async function compileAuto2000Module(wasmSource = null) {
  if (!cachedModulePromise || wasmSource !== null) {
    const p = compileWasm(wasmSource);
    if (wasmSource === null) cachedModulePromise = p;
    return p;
  }
  return cachedModulePromise;
}

/**
 * Classify stability from AUTO2000 eigenvalues (IPS=1) or Floquet multipliers (IPS=2).
 * For IPS=2, excludes the single trivial Floquet multiplier closest to (1, 0).
 */
function classifyStability(ips, ev, isw = 1) {
  if (Math.abs(isw) === 2) return null;
  if (!ev || ev.length === 0) return null;
  if (ips === 1 || ips === 0) {
    return ev.every((e) => e.r < 1e-6);
  }
  if (ips === 2) {
    let trivialIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < ev.length; i++) {
      const d = Math.hypot(ev[i].r - 1.0, ev[i].i);
      if (d < bestDist) {
        bestDist = d;
        trivialIdx = i;
      }
    }
    for (let i = 0; i < ev.length; i++) {
      if (i === trivialIdx) continue;
      if (ev[i].mag > 1.0 + 1e-5) return false;
    }
    return true;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Dense Linear Algebra & Codim-2 Test Functions (CP, BT, GH, ZH)
// ---------------------------------------------------------------------------

function evalFunc(func, u, p0, p1, n) {
  const f = new Float64Array(n);
  const par = new Float64Array([p0, p1]);
  func(u, par, f);
  return f;
}

function computeJacobian(func, x, p0, p1, n) {
  const J = Array.from({ length: n }, () => new Float64Array(n));
  const xp = new Float64Array(x);
  const xm = new Float64Array(x);
  const par = new Float64Array([p0, p1]);
  const fp = new Float64Array(n);
  const fm = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const h = 1e-6 * Math.max(1.0, Math.abs(x[j]));
    xp[j] = x[j] + h;
    xm[j] = x[j] - h;
    func(xp, par, fp);
    func(xm, par, fm);
    const inv2h = 0.5 / h;
    for (let i = 0; i < n; i++) {
      J[i][j] = (fp[i] - fm[i]) * inv2h;
    }
    xp[j] = x[j];
    xm[j] = x[j];
  }
  return J;
}

function solveLinearSystem(A, b, n) {
  const M = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n + 1);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n] = b[i];
    return row;
  });
  for (let k = 0; k < n; k++) {
    let piv = k;
    let maxV = Math.abs(M[k][k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(M[i][k]);
      if (v > maxV) {
        maxV = v;
        piv = i;
      }
    }
    if (piv !== k) {
      const tmp = M[k];
      M[k] = M[piv];
      M[piv] = tmp;
    }
    const diag = M[k][k];
    if (Math.abs(diag) < 1e-15) {
      M[k][k] = diag >= 0 ? 1e-15 : -1e-15;
    }
    for (let i = k + 1; i < n; i++) {
      const factor = M[i][k] / M[k][k];
      if (factor === 0) continue;
      for (let j = k + 1; j <= n; j++) {
        M[i][j] -= factor * M[k][j];
      }
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }
  return x;
}

/**
 * Compute unit nullvector of a nearly-singular n x n matrix via complete-pivoting LU.
 */
function nullVectorCompletePivoting(A, n) {
  const M = Array.from({ length: n }, (_, i) => Float64Array.from(A[i]));
  const colPerm = Array.from({ length: n }, (_, j) => j);
  for (let k = 0; k < n - 1; k++) {
    let pivR = k;
    let pivC = k;
    let maxV = 0;
    for (let i = k; i < n; i++) {
      for (let j = k; j < n; j++) {
        const v = Math.abs(M[i][j]);
        if (v > maxV) {
          maxV = v;
          pivR = i;
          pivC = j;
        }
      }
    }
    if (maxV < 1e-15) break;
    if (pivR !== k) {
      const tmp = M[k];
      M[k] = M[pivR];
      M[pivR] = tmp;
    }
    if (pivC !== k) {
      for (let i = 0; i < n; i++) {
        const t = M[i][k];
        M[i][k] = M[i][pivC];
        M[i][pivC] = t;
      }
      const tc = colPerm[k];
      colPerm[k] = colPerm[pivC];
      colPerm[pivC] = tc;
    }
    for (let i = k + 1; i < n; i++) {
      const factor = M[i][k] / M[k][k];
      for (let j = k + 1; j < n; j++) {
        M[i][j] -= factor * M[k][j];
      }
    }
  }
  const y = new Float64Array(n);
  y[n - 1] = 1.0;
  for (let i = n - 2; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * y[j];
    }
    const diag = M[i][i];
    y[i] = Math.abs(diag) > 1e-14 ? sum / diag : 0.0;
  }
  const v = new Float64Array(n);
  let norm2 = 0;
  for (let j = 0; j < n; j++) {
    v[colPerm[j]] = y[j];
    norm2 += y[j] * y[j];
  }
  const invNorm = norm2 > 0 ? 1.0 / Math.sqrt(norm2) : 1.0;
  for (let j = 0; j < n; j++) v[j] *= invNorm;
  return v;
}

function alignVec(v, prev) {
  if (!prev) return v;
  let d = 0;
  for (let i = 0; i < v.length; i++) d += v[i] * prev[i];
  if (d < 0) {
    for (let i = 0; i < v.length; i++) v[i] = -v[i];
  }
  return v;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Compute all complex eigenvalues of a real n x n matrix via Hessenberg + Francis QR.
 */
function realMatrixEigenvalues(A, n) {
  if (n === 1) return [{ r: A[0][0], i: 0 }];
  if (n === 2) {
    const tr = A[0][0] + A[1][1];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const disc = tr * tr - 4 * det;
    if (disc >= 0) {
      const sq = Math.sqrt(disc);
      return [{ r: 0.5 * (tr + sq), i: 0 }, { r: 0.5 * (tr - sq), i: 0 }];
    }
    const sq = Math.sqrt(-disc);
    return [{ r: 0.5 * tr, i: 0.5 * sq }, { r: 0.5 * tr, i: -0.5 * sq }];
  }

  // Copy matrix
  const H = Array.from({ length: n }, (_, i) => Float64Array.from(A[i]));

  // Reduce to upper Hessenberg via Householder reflections
  for (let k = 0; k < n - 2; k++) {
    let scale = 0;
    for (let i = k + 1; i < n; i++) scale += Math.abs(H[i][k]);
    if (scale === 0) continue;
    const len = n - k - 1;
    const u = new Float64Array(len);
    let norm2 = 0;
    for (let i = 0; i < len; i++) {
      u[i] = H[k + 1 + i][k] / scale;
      norm2 += u[i] * u[i];
    }
    let norm = Math.sqrt(norm2);
    if (u[0] >= 0) norm = -norm;
    u[0] -= norm;
    let u2 = 0;
    for (let i = 0; i < len; i++) u2 += u[i] * u[i];
    if (u2 < 1e-30) continue;
    const invU2 = 2.0 / u2;

    for (let j = k; j < n; j++) {
      let s = 0;
      for (let i = 0; i < len; i++) s += u[i] * H[k + 1 + i][j];
      s *= invU2;
      for (let i = 0; i < len; i++) H[k + 1 + i][j] -= s * u[i];
    }
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < len; j++) s += H[i][k + 1 + j] * u[j];
      s *= invU2;
      for (let j = 0; j < len; j++) H[i][k + 1 + j] -= s * u[j];
    }
  }

  // Francis double-shift QR
  const ev = [];
  let nn = n - 1;
  let iter = 0;
  let totalIter = 0;
  const maxTotalIter = 60 * n;
  let normH = 0;
  for (let i = 0; i < n; i++) {
    for (let j = Math.max(0, i - 1); j < n; j++) normH += Math.abs(H[i][j]);
  }
  const eps = 1e-13;

  while (nn >= 0 && totalIter < maxTotalIter) {
    let l = nn;
    while (l > 0) {
      const s = Math.abs(H[l - 1][l - 1]) + Math.abs(H[l][l]);
      const tol = eps * (s > 0 ? s : normH || 1);
      if (Math.abs(H[l][l - 1]) <= tol) {
        H[l][l - 1] = 0;
        break;
      }
      l--;
    }

    if (l === nn) {
      ev.push({ r: H[nn][nn], i: 0 });
      nn--;
      iter = 0;
    } else if (l === nn - 1) {
      const p = 0.5 * (H[nn - 1][nn - 1] - H[nn][nn]);
      const q = p * p + H[nn][nn - 1] * H[nn - 1][nn];
      const tr = H[nn][nn] + p;
      if (q >= 0) {
        const z = Math.sqrt(q);
        ev.push({ r: tr + z, i: 0 });
        ev.push({ r: tr - z, i: 0 });
      } else {
        const z = Math.sqrt(-q);
        ev.push({ r: tr, i: z });
        ev.push({ r: tr, i: -z });
      }
      nn -= 2;
      iter = 0;
    } else {
      iter++;
      totalIter++;
      let s, t;
      if (iter % 10 === 0) {
        const ad = Math.abs(H[nn][nn - 1]) + Math.abs(H[nn - 1][nn - 2]);
        s = 1.5 * ad;
        t = ad * ad;
      } else {
        const a = H[nn - 1][nn - 1];
        const b = H[nn][nn];
        s = a + b;
        t = a * b - H[nn - 1][nn] * H[nn][nn - 1];
      }
      let m = nn - 2;
      let p = 0, q = 0, r = 0;
      for (; m >= l; m--) {
        const hmm = H[m][m];
        const r1 = H[m + 1][m];
        p = hmm * (hmm - s) + t + H[m][m + 1] * r1;
        q = r1 * (hmm + H[m + 1][m + 1] - s);
        r = r1 * H[m + 2][m + 1];
        if (m === l) break;
        const lhs = Math.abs(H[m][m - 1]) * (Math.abs(q) + Math.abs(r));
        const rhs = Math.abs(p) * (Math.abs(H[m - 1][m - 1]) + Math.abs(hmm) + Math.abs(H[m + 1][m + 1]));
        if (lhs <= eps * rhs) break;
      }
      for (let i = m + 2; i <= nn; i++) {
        H[i][i - 2] = 0;
        if (i > m + 2) H[i][i - 3] = 0;
      }
      for (let k = m; k < nn; k++) {
        const notLast = k !== nn - 1;
        if (k !== m) {
          p = H[k][k - 1];
          q = H[k + 1][k - 1];
          r = notLast ? H[k + 2][k - 1] : 0;
        }
        const x = Math.abs(p) + Math.abs(q) + Math.abs(r);
        if (x === 0) continue;
        p /= x;
        q /= x;
        r /= x;
        let ss = Math.sqrt(p * p + q * q + r * r);
        if (p < 0) ss = -ss;
        if (k !== m) H[k][k - 1] = -ss * x;
        else if (l !== m) H[k][k - 1] = -H[k][k - 1];
        p += ss;
        const xx = p / ss;
        const yy = q / ss;
        const zz = r / ss;
        q /= p;
        r /= p;
        for (let j = k; j < n; j++) {
          let pp = H[k][j] + q * H[k + 1][j];
          if (notLast) {
            pp += r * H[k + 2][j];
            H[k + 2][j] -= pp * zz;
          }
          H[k + 1][j] -= pp * yy;
          H[k][j] -= pp * xx;
        }
        const lim = Math.min(nn, k + 3);
        for (let i = 0; i <= lim; i++) {
          let pp = xx * H[i][k] + yy * H[i][k + 1];
          if (notLast) {
            pp += zz * H[i][k + 2];
            H[i][k + 2] -= pp * r;
          }
          H[i][k + 1] -= pp * q;
          H[i][k] -= pp;
        }
      }
    }
  }
  return ev;
}

function deriv2a(func, x, p0, p1, u, n) {
  let maxX = 1.0;
  for (let i = 0; i < n; i++) maxX = Math.max(maxX, Math.abs(x[i]));
  const h = 1e-4 * maxX;
  const f0 = evalFunc(func, x, p0, p1, n);
  const xp = new Float64Array(n);
  const xm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xp[i] = x[i] + h * u[i];
    xm[i] = x[i] - h * u[i];
  }
  const fp = evalFunc(func, xp, p0, p1, n);
  const fm = evalFunc(func, xm, p0, p1, n);
  const out = new Float64Array(n);
  const invH2 = 1.0 / (h * h);
  for (let i = 0; i < n; i++) {
    out[i] = (fp[i] - 2 * f0[i] + fm[i]) * invH2;
  }
  return out;
}

function deriv2b(func, x, p0, p1, u, v, n) {
  let maxX = 1.0;
  for (let i = 0; i < n; i++) maxX = Math.max(maxX, Math.abs(x[i]));
  const h = 1e-4 * maxX;
  const xpp = new Float64Array(n);
  const xpm = new Float64Array(n);
  const xmp = new Float64Array(n);
  const xmm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xpp[i] = x[i] + h * (u[i] + v[i]);
    xpm[i] = x[i] + h * (u[i] - v[i]);
    xmp[i] = x[i] - h * (u[i] - v[i]);
    xmm[i] = x[i] - h * (u[i] + v[i]);
  }
  const fpp = evalFunc(func, xpp, p0, p1, n);
  const fpm = evalFunc(func, xpm, p0, p1, n);
  const fmp = evalFunc(func, xmp, p0, p1, n);
  const fmm = evalFunc(func, xmm, p0, p1, n);
  const out = new Float64Array(n);
  const inv4H2 = 0.25 / (h * h);
  for (let i = 0; i < n; i++) {
    out[i] = (fpp[i] - fpm[i] - fmp[i] + fmm[i]) * inv4H2;
  }
  return out;
}

function deriv3a(func, x, p0, p1, u, n) {
  let maxX = 1.0;
  for (let i = 0; i < n; i++) maxX = Math.max(maxX, Math.abs(x[i]));
  const h = 5e-3 * maxX;
  const xp1 = new Float64Array(n);
  const xm1 = new Float64Array(n);
  const xp2 = new Float64Array(n);
  const xm2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xp1[i] = x[i] + h * u[i];
    xm1[i] = x[i] - h * u[i];
    xp2[i] = x[i] + 2 * h * u[i];
    xm2[i] = x[i] - 2 * h * u[i];
  }
  const fp1 = evalFunc(func, xp1, p0, p1, n);
  const fm1 = evalFunc(func, xm1, p0, p1, n);
  const fp2 = evalFunc(func, xp2, p0, p1, n);
  const fm2 = evalFunc(func, xm2, p0, p1, n);
  const out = new Float64Array(n);
  const inv2H3 = 0.5 / (h * h * h);
  for (let i = 0; i < n; i++) {
    out[i] = (fp2[i] - 2 * fp1[i] + 2 * fm1[i] - fm2[i]) * inv2H3;
  }
  return out;
}

/**
 * First Lyapunov coefficient l1 at a Hopf point (x, p0, p1) with frequency omega > 0.
 */
function computeFirstLyapunov(func, x, p0, p1, omega, n, J) {
  if (!(omega > 1e-6)) return null;
  // Build 2n x 2n real matrices for (J - i*omega*I)q = 0 and (J^T + i*omega*I)p = 0
  const Mq = Array.from({ length: 2 * n }, () => new Float64Array(2 * n));
  const Mp = Array.from({ length: 2 * n }, () => new Float64Array(2 * n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Mq[i][j] = J[i][j];
      Mq[n + i][n + j] = J[i][j];
      Mp[i][j] = J[j][i];
      Mp[n + i][n + j] = J[j][i];
    }
    Mq[i][n + i] = omega;
    Mq[n + i][i] = -omega;
    Mp[i][n + i] = -omega;
    Mp[n + i][i] = omega;
  }
  const qVec = nullVectorCompletePivoting(Mq, 2 * n);
  const qR = qVec.slice(0, n);
  const qI = qVec.slice(n, 2 * n);
  const pVec = nullVectorCompletePivoting(Mp, 2 * n);
  let pR = pVec.slice(0, n);
  let pI = pVec.slice(n, 2 * n);

  // Normalize so <p, q> = (pR - i pI)^T (qR + i qI) = 1
  const rePQ = dot(pR, qR) + dot(pI, qI);
  const imPQ = dot(pR, qI) - dot(pI, qR);
  const denom = rePQ * rePQ + imPQ * imPQ;
  if (!(denom > 1e-20)) return null;
  // Divide p by conj(rePQ + i imPQ) = rePQ - i imPQ
  // Wait: <p/z, q> = (1/conj(z)) <p, q>. So choose z = conj(<p, q>) => pNew = p / z
  const newPR = new Float64Array(n);
  const newPI = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    newPR[i] = (pR[i] * rePQ - pI[i] * imPQ) / denom;
    newPI[i] = (pI[i] * rePQ + pR[i] * imPQ) / denom;
  }
  pR = newPR;
  pI = newPI;

  const a = deriv2a(func, x, p0, p1, qR, n);
  const b = deriv2a(func, x, p0, p1, qI, n);
  const c = deriv2b(func, x, p0, p1, qR, qI, n);

  // r = J^{-1} B(q, conj(q)) = J^{-1}(a + b)
  const abSum = new Float64Array(n);
  for (let i = 0; i < n; i++) abSum[i] = a[i] + b[i];
  const r = solveLinearSystem(J, abSum, n);

  // (2i*omega*I - J) s = B(q, q) = (a - b) + i*(2*c)
  const Ms = Array.from({ length: 2 * n }, () => new Float64Array(2 * n));
  const rhsS = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      Ms[i][j] = -J[i][j];
      Ms[n + i][n + j] = -J[i][j];
    }
    Ms[i][n + i] = -2 * omega;
    Ms[n + i][i] = 2 * omega;
    rhsS[i] = a[i] - b[i];
    rhsS[n + i] = 2 * c[i];
  }
  const sVec = solveLinearSystem(Ms, rhsS, 2 * n);
  const sR = sVec.slice(0, n);
  const sI = sVec.slice(n, 2 * n);

  // Sigma = Re <p, B(q, r)>
  const bQrR = deriv2b(func, x, p0, p1, qR, r, n);
  const bQrI = deriv2b(func, x, p0, p1, qI, r, n);
  const sigma = dot(pR, bQrR) + dot(pI, bQrI);

  // Delta = Re <p, B(conj(q), s)>
  const bqRsR = deriv2b(func, x, p0, p1, qR, sR, n);
  const bqIsI = deriv2b(func, x, p0, p1, qI, sI, n);
  const bqRsI = deriv2b(func, x, p0, p1, qR, sI, n);
  const bqIsR = deriv2b(func, x, p0, p1, qI, sR, n);
  const delta = dot(pR, bqRsR) + dot(pR, bqIsI) + dot(pI, bqRsI) - dot(pI, bqIsR);

  // Gamma = Re <p, C(q, q, conj(q))>
  const c1 = deriv3a(func, x, p0, p1, qR, n);
  const c2 = deriv3a(func, x, p0, p1, qI, n);
  const qPlus = new Float64Array(n);
  const qMinus = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    qPlus[i] = qR[i] + qI[i];
    qMinus[i] = qR[i] - qI[i];
  }
  const cPlus = deriv3a(func, x, p0, p1, qPlus, n);
  const cMinus = deriv3a(func, x, p0, p1, qMinus, n);
  // C(qR, qI, qI) = (cPlus + cMinus - 2*c1)/6; C(qI, qR, qR) = (cPlus - cMinus - 2*c2)/6
  const reC = new Float64Array(n);
  const imC = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    reC[i] = c1[i] + (cPlus[i] + cMinus[i] - 2 * c1[i]) / 6.0;
    imC[i] = c2[i] + (cPlus[i] - cMinus[i] - 2 * c2[i]) / 6.0;
  }
  const gamma = dot(pR, reC) + dot(pI, imC);

  const l1 = (gamma - 2 * sigma + delta) / (2 * omega);
  return Number.isFinite(l1) ? l1 : null;
}

/**
 * Evaluate codim-2 test functions along LP2 or HB2 and refine/label CP, BT, GH, ZH.
 */
function detectCodim2SpecialPoints({ mod, func, ndim, points, specialPoints, restartSp }) {
  const isLP2 = restartSp.itp === 2 || Math.trunc(Math.abs(restartSp.itp) / 10) === 2;
  const isHB2 = restartSp.itp === 3 || Math.trunc(Math.abs(restartSp.itp) / 10) === 3;
  if ((!isLP2 && !isHB2) || points.length < 2) return specialPoints;

  const numU = points.length;
  const diag = [];
  let prevV = null;
  let prevW = null;

  for (let k = 0; k < numU; k++) {
    const x = Float64Array.from(points[k].u);
    const p0 = points[k].par[0];
    const p1 = points[k].par[1];
    const J = computeJacobian(func, x, p0, p1, ndim);
    const evs = realMatrixEigenvalues(J, ndim);

    if (isLP2) {
      let v = new Float64Array(ndim);
      let vNorm2 = 0;
      for (let j = 0; j < ndim; j++) {
        v[j] = mod._auto2000_u(k, ndim + j);
        vNorm2 += v[j] * v[j];
      }
      if (vNorm2 > 1e-20) {
        const inv = 1.0 / Math.sqrt(vNorm2);
        for (let j = 0; j < ndim; j++) v[j] *= inv;
      } else {
        v = nullVectorCompletePivoting(J, ndim);
      }
      v = alignVec(v, prevV);
      prevV = v;

      const JT = Array.from({ length: ndim }, (_, i) => {
        const row = new Float64Array(ndim);
        for (let j = 0; j < ndim; j++) row[j] = J[j][i];
        return row;
      });
      let w = alignVec(nullVectorCompletePivoting(JT, ndim), prevW);
      prevW = w;

      const psiBT = dot(w, v);
      const d2v = deriv2a(func, x, p0, p1, v, ndim);
      const psiCP = 0.5 * dot(w, d2v);

      // Zero-Hopf check for ndim >= 3: non-zero eigenvalues with Re(lambda) > 0
      let zhProd = 1.0;
      let complexReList = [];
      if (ndim >= 3) {
        // Exclude eigenvalue closest to 0
        let zeroIdx = 0;
        let bestMag = Infinity;
        for (let i = 0; i < evs.length; i++) {
          const m = Math.hypot(evs[i].r, evs[i].i);
          if (m < bestMag) {
            bestMag = m;
            zeroIdx = i;
          }
        }
        const rem = evs.filter((_, idx) => idx !== zeroIdx);
        for (let i = 0; i < rem.length; i++) {
          if (rem[i].i > 1e-5) complexReList.push(rem[i].r);
        }
        // Product of pairwise sums of remaining eigenvalues (changes sign when a complex pair crosses Re=0!)
        for (let i = 0; i < rem.length; i++) {
          for (let j = i + 1; j < rem.length; j++) {
            zhProd *= rem[i].r + rem[j].r - (rem[i].i - rem[j].i === 0 ? 0 : 0);
          }
        }
      }
      diag.push({ x, p0, p1, psiBT, psiCP, evs, complexReList, zhProd });
    } else if (isHB2) {
      const rom = mod._auto2000_u(k, 3 * ndim);
      const omega = Math.abs(rom) > 1e-12 ? 1.0 / rom : 0;
      const absOmega = Math.abs(omega);
      const l1 = absOmega > 1e-4 ? computeFirstLyapunov(func, x, p0, p1, absOmega, ndim, J) : null;

      // Check real non-Hopf eigenvalues for ZH (det(J) or product of non-Hopf real parts)
      let nonHopfDet = 1.0;
      let nNonHopfRealPos = 0;
      if (ndim >= 3) {
        // Exclude the pair closest to +/- i*absOmega
        const used = new Set();
        for (const sign of [1, -1]) {
          let bestIdx = -1;
          let bestD = Infinity;
          for (let i = 0; i < evs.length; i++) {
            if (used.has(i)) continue;
            const d = Math.hypot(evs[i].r, evs[i].i - sign * absOmega);
            if (d < bestD) {
              bestD = d;
              bestIdx = i;
            }
          }
          if (bestIdx >= 0) used.add(bestIdx);
        }
        for (let i = 0; i < evs.length; i++) {
          if (used.has(i)) continue;
          if (Math.abs(evs[i].i) < 1e-6) {
            nonHopfDet *= evs[i].r;
            if (evs[i].r > 0) nNonHopfRealPos++;
          }
        }
      }
      diag.push({ x, p0, p1, omega, absOmega, l1, evs, nonHopfDet, nNonHopfRealPos });
    }
  }

  let nextLab = specialPoints.reduce((mx, sp) => Math.max(mx, sp.lab || 0), 0) + 1;
  const templatePar = specialPoints[0]?.par ? [...specialPoints[0].par] : new Array(11).fill(0);

  const upsertCodim2 = (ty, interpX, p0, p1, omegaOpt = null) => {
    // Check if an existing non-EP/non-UZ AUTO2000 special point is near (p0, p1)
    let bestSp = null;
    let bestDist = Infinity;
    for (const sp of specialPoints) {
      if (sp.ty === 'EP' || sp.ty === 'MX' || sp.ty === 'UZ' || sp.ty === 'RG') continue;
      if (sp.ty === 'CP' || sp.ty === 'BT' || sp.ty === 'GH' || sp.ty === 'ZH') continue;
      const d = Math.hypot(sp.par1 - p0, sp.par2 - p1);
      if (d < bestDist) {
        bestDist = d;
        bestSp = sp;
      }
    }
    const tol = 0.05 * Math.max(1.0, Math.hypot(p0, p1));
    if (bestSp && bestDist < tol) {
      bestSp.ty = ty;
      return;
    }
    const parArr = [...templatePar];
    parArr[0] = p0;
    parArr[1] = p1;
    if (omegaOpt !== null && omegaOpt > 1e-6) {
      parArr[10] = (2 * Math.PI) / omegaOpt;
    }
    const itpMap = isLP2
      ? { CP: 22, BT: 23, ZH: 23 }
      : { GH: 32, BT: 31, ZH: 33 };
    specialPoints.push({
      itp: itpMap[ty] ?? 0,
      ty,
      lab: nextLab++,
      par1: p0,
      par2: p1,
      par: parArr,
      u1Based: [0, ...Array.from(interpX)],
    });
  };

  const interpPoint = (k0, k1, alpha) => {
    const t = Math.min(1, Math.max(0, alpha));
    const x = new Float64Array(ndim);
    for (let j = 0; j < ndim; j++) {
      x[j] = (1 - t) * diag[k0].x[j] + t * diag[k1].x[j];
    }
    const p0 = (1 - t) * diag[k0].p0 + t * diag[k1].p0;
    const p1 = (1 - t) * diag[k0].p1 + t * diag[k1].p1;
    return { x, p0, p1 };
  };

  if (isLP2) {
    for (let k = 1; k < numU; k++) {
      const a = diag[k - 1];
      const b = diag[k];
      const cpCross = a.psiCP * b.psiCP < 0 || (Math.abs(b.psiCP) < 1e-9 && Math.abs(a.psiCP) > 1e-9);
      const btCross = a.psiBT * b.psiBT < 0 || (Math.abs(b.psiBT) < 1e-9 && Math.abs(a.psiBT) > 1e-9);
      if (cpCross) {
        const alpha = Math.abs(b.psiCP - a.psiCP) > 1e-15 ? -a.psiCP / (b.psiCP - a.psiCP) : 0.5;
        const { x, p0, p1 } = interpPoint(k - 1, k, alpha);
        upsertCodim2('CP', x, p0, p1);
      } else if (btCross) {
        const alpha = Math.abs(b.psiBT - a.psiBT) > 1e-15 ? -a.psiBT / (b.psiBT - a.psiBT) : 0.5;
        const { x, p0, p1 } = interpPoint(k - 1, k, alpha);
        upsertCodim2('BT', x, p0, p1);
      }
      if (ndim >= 3 && a.complexReList.length > 0 && b.complexReList.length > 0) {
        if (a.zhProd * b.zhProd < 0) {
          const alpha = Math.abs(b.zhProd - a.zhProd) > 1e-15 ? -a.zhProd / (b.zhProd - a.zhProd) : 0.5;
          const { x, p0, p1 } = interpPoint(k - 1, k, alpha);
          upsertCodim2('ZH', x, p0, p1);
        }
      }
    }
    // Check if AUTO2000 emitted an unmapped BP/LP/HB on LP2 without a sign cross in interior
    for (const sp of specialPoints) {
      if (sp.ty === 'CP' || sp.ty === 'BT' || sp.ty === 'GH' || sp.ty === 'ZH') continue;
      if (sp.itp === 22) sp.ty = 'CP';
      else if (sp.itp === 21 || sp.itp === 23) {
        // Disambiguate BT vs ZH via Jacobian at sp
        const xSp = Float64Array.from(sp.u1Based.slice(1, 1 + ndim));
        const Jsp = computeJacobian(func, xSp, sp.par1, sp.par2, ndim);
        const evSp = realMatrixEigenvalues(Jsp, ndim);
        const nearZero = evSp.filter((e) => Math.hypot(e.r, e.i) < 1e-2).length;
        const imagPair = ndim >= 3 && evSp.some((e) => Math.abs(e.r) < 1e-2 && Math.abs(e.i) > 1e-2);
        if (nearZero >= 2) sp.ty = 'BT';
        else if (imagPair) sp.ty = 'ZH';
      }
    }
  } else if (isHB2) {
    for (let k = 1; k < numU; k++) {
      const a = diag[k - 1];
      const b = diag[k];
      if (a.omega * b.omega < 0) {
        const alpha = Math.abs(b.omega - a.omega) > 1e-15 ? -a.omega / (b.omega - a.omega) : 0.5;
        const { x, p0, p1 } = interpPoint(k - 1, k, alpha);
        upsertCodim2('BT', x, p0, p1);
      }
      if (a.l1 !== null && b.l1 !== null && a.l1 * b.l1 < 0) {
        const alpha = Math.abs(b.l1 - a.l1) > 1e-15 ? -a.l1 / (b.l1 - a.l1) : 0.5;
        const { x, p0, p1 } = interpPoint(k - 1, k, alpha);
        const om = (1 - alpha) * a.absOmega + alpha * b.absOmega;
        upsertCodim2('GH', x, p0, p1, om);
      }
      if (ndim >= 3 && a.nNonHopfRealPos !== b.nNonHopfRealPos && a.absOmega > 1e-4 && b.absOmega > 1e-4) {
        const alpha = Math.abs(b.nonHopfDet - a.nonHopfDet) > 1e-15 ? -a.nonHopfDet / (b.nonHopfDet - a.nonHopfDet) : 0.5;
        const { x, p0, p1 } = interpPoint(k - 1, k, alpha);
        const om = (1 - alpha) * a.absOmega + alpha * b.absOmega;
        upsertCodim2('ZH', x, p0, p1, om);
      }
    }
    // Check if HB2 terminated near BT (omega -> 0) without crossing 0
    if (!specialPoints.some((sp) => sp.ty === 'BT')) {
      const last = diag[numU - 1];
      const first = diag[0];
      let peakOmega = 0;
      for (const d of diag) peakOmega = Math.max(peakOmega, d.absOmega);
      const btCandidate = last.absOmega < 0.15 && last.absOmega < 0.15 * peakOmega
        ? last
        : (first.absOmega < 0.15 && first.absOmega < 0.15 * peakOmega ? first : null);
      if (btCandidate) {
        const Jc = computeJacobian(func, btCandidate.x, btCandidate.p0, btCandidate.p1, ndim);
        const evC = realMatrixEigenvalues(Jc, ndim);
        const smallCount = evC.filter((e) => Math.hypot(e.r, e.i) < 0.2).length;
        if (smallCount >= 2) {
          upsertCodim2('BT', btCandidate.x, btCandidate.p0, btCandidate.p1);
        }
      }
    }
    for (const sp of specialPoints) {
      if (sp.ty === 'CP' || sp.ty === 'BT' || sp.ty === 'GH' || sp.ty === 'ZH') continue;
      if (sp.itp === 31 || sp.itp === 33) {
        const xSp = Float64Array.from(sp.u1Based.slice(1, 1 + ndim));
        const Jsp = computeJacobian(func, xSp, sp.par1, sp.par2, ndim);
        const evSp = realMatrixEigenvalues(Jsp, ndim);
        const nearZero = evSp.filter((e) => Math.hypot(e.r, e.i) < 1e-2).length;
        const imagPair = ndim >= 3 && evSp.some((e) => Math.abs(e.r) < 1e-2 && Math.abs(e.i) > 1e-2);
        if (nearZero >= 2) sp.ty = 'BT';
        else if (imagPair) sp.ty = 'ZH';
      }
    }
  }

  return specialPoints;
}

// ---------------------------------------------------------------------------
// Main Runner
// ---------------------------------------------------------------------------

/**
 * Run an isolated AUTO2000 continuation in a fresh WebAssembly.Instance.
 *
 * Contract: Only pass 1 or 2 active continuation parameters in par[0] (and par[1] for 2D).
 * Keep all other fixed model parameters in the JS closure around `func` so they never
 * collide with AUTO's reserved PAR(11) period slot (par[10]).
 */
async function runAuto2000({
  ndim,
  ips = 1,
  irs = 0,
  ilp = 1,
  isw = 1,
  isp = 2,
  nmx = 200,
  npr = 200,
  ntst = 50,
  ncol = 4,
  ds = 0.05,
  dsmin = 1e-3,
  dsmax = 0.2,
  rl0 = -1e30,
  rl1 = 1e30,
  icp = null,
  iuz = [],
  vuz = [],
  restartSp = null,
  wasmModule = null,
  wasmUrl = null,
  locateFile = null,
  func,
  stpnt = null,
  _skipStage1 = false,
}) {
  if (typeof func !== 'function') {
    throw new TypeError('runAuto2000 requires a func(u, par, f) callback');
  }

  // Enforce PAR(11) collision safety: user active parameters live in par[0] (and par[1])
  const activeIcp = icp ?? (ips === 2 ? (isw === 2 ? [0, 1] : [0, 10]) : isw === 2 ? [0, 1] : [0]);
  if (activeIcp.some((idx) => idx < 0 || (idx >= 10 && !(ips === 2 && idx === 10)))) {
    throw new RangeError(`Invalid active icp=${JSON.stringify(activeIcp)}; keep user params in [0, 1]`);
  }

  // Handle two-stage AUTO2000 startup for 2D LC bifurcations (LPC=5, BPC=6, PD=7, TR=8)
  if (!_skipStage1 && ips === 2 && Math.abs(isw) === 2 && restartSp && [5, 6, 7, 8].includes(restartSp.itp)) {
    const stage1 = await runAuto2000({
      ndim,
      ips: 2,
      irs,
      ilp: 0,
      isw: 2,
      isp: 0,
      nmx: 10,
      npr: 10,
      ntst,
      ncol,
      ds: Math.abs(ds) || 0.05,
      dsmin,
      dsmax,
      rl0: -1e30,
      rl1: 1e30,
      icp: activeIcp,
      restartSp,
      wasmModule,
      wasmUrl,
      locateFile,
      func,
      _skipStage1: true,
    });
    if (stage1.error) return stage1;
    const targetPrefix = restartSp.itp * 10 + 9;
    const prepSp = stage1.specialPoints.find((sp) => sp.itp === targetPrefix)
      ?? stage1.specialPoints[stage1.specialPoints.length - 1];
    if (!prepSp) {
      return {
        rc: -1,
        error: new Error(`Failed to generate 2D LC starting point for itp=${restartSp.itp}`),
        points: [],
        specialPoints: [],
        instantiateMs: stage1.instantiateMs,
        runMs: stage1.runMs,
      };
    }
    return runAuto2000({
      ndim,
      ips: 2,
      irs,
      ilp,
      isw,
      isp,
      nmx,
      npr,
      ntst,
      ncol,
      ds,
      dsmin,
      dsmax,
      rl0,
      rl1,
      icp: activeIcp,
      iuz,
      vuz,
      restartSp: prepSp,
      wasmModule,
      wasmUrl,
      locateFile,
      func,
      stpnt,
      _skipStage1: true,
    });
  }

  // Fixed-period 2D LC continuation (e.g. HOM2 from high-period UZ/EP/RG):
  // AUTO2000 requires ips=2, isw=1 with icp=[0, 1] (holding PAR(11) fixed).
  let effectiveIsw = isw;
  if (
    ips === 2 &&
    Math.abs(isw) === 2 &&
    restartSp &&
    ![5, 6, 7, 8].includes(restartSp.itp) &&
    ![5, 6, 7, 8].includes(Math.trunc(Math.abs(restartSp.itp) / 10))
  ) {
    effectiveIsw = 1;
  }

  const compiled = await compileAuto2000Module(wasmModule ?? wasmUrl);
  const tInst0 = performance.now();

  const mod = await createAuto2000Core({
    noInitialRun: true,
    print: () => {},
    ...(locateFile ? { locateFile } : {}),
    instantiateWasm(imports, receiveInstance) {
      const instance = new WebAssembly.Instance(compiled, imports);
      receiveInstance(instance, compiled);
      return instance.exports;
    },
  });
  const instantiateMs = performance.now() - tInst0;

  let cbErr = null;
  mod._autoCallbacks = {
    func: (n, uPtr, _icpPtr, parPtr, _ijac, fPtr) => {
      try {
        const buf = mod.HEAPF64.buffer;
        const f = new Float64Array(buf, fPtr, n);
        func(new Float64Array(buf, uPtr, n), new Float64Array(buf, parPtr, 2), f);
        for (let i = 0; i < n; i++) {
          if (!Number.isFinite(f[i])) {
            throw new Error(`Non-finite f[${i}]=${f[i]} returned by JS func`);
          }
        }
      } catch (err) {
        cbErr = err;
        throw err;
      }
    },
    stpnt: (n, t, uPtr, parPtr) => {
      if (!stpnt) return;
      try {
        const buf = mod.HEAPF64.buffer;
        stpnt(new Float64Array(buf, uPtr, n), new Float64Array(buf, parPtr, 2), t);
      } catch (err) {
        cbErr = err;
        throw err;
      }
    },
  };

  const npar = 11;
  const icpPtr = mod._malloc(activeIcp.length * 4);
  mod.HEAP32.set(activeIcp, icpPtr >> 2);
  let iuzPtr = 0;
  let vuzPtr = 0;
  if (iuz.length > 0) {
    iuzPtr = mod._malloc(iuz.length * 4);
    vuzPtr = mod._malloc(vuz.length * 8);
    mod.HEAP32.set(iuz, iuzPtr >> 2);
    mod.HEAPF64.set(vuz, vuzPtr >> 3);
  }

  const effectiveIrs = restartSp ? restartSp.lab : irs;
  mod._auto2000_init(
    ndim, npar, ips, effectiveIrs, ilp, effectiveIsw, isp,
    nmx, npr, ntst, ncol, ds, dsmin, dsmax, rl0, rl1,
    activeIcp.length, icpPtr, iuz.length, iuzPtr, vuzPtr,
  );

  if (restartSp) {
    const uPtr = mod._malloc(restartSp.u1Based.length * 8);
    const parPtr = mod._malloc(restartSp.par.length * 8);
    mod.HEAPF64.set(restartSp.u1Based, uPtr >> 3);
    mod.HEAPF64.set(restartSp.par, parPtr >> 3);

    if (restartSp.ups && restartSp.ups.length > 0) {
      const spNtst = restartSp.ntst ?? ntst;
      const spNcol = restartSp.ncol ?? ncol;
      const ntpl = restartSp.ntpl ?? (spNtst * spNcol + 1);
      const nar = restartSp.nar ?? (Array.isArray(restartSp.ups[0]) ? restartSp.ups[0].length + 1 : restartSp.ups.length / ntpl);
      const spNdim = nar - 1;
      const spIcp = restartSp.icp ?? activeIcp;
      const spNfpr = restartSp.nfpr ?? spIcp.length;
      const spRldot = restartSp.rldot ?? new Array(spNfpr).fill(0);

      // Pack upsFlat as ntpl * nar (col 0 = t, cols 1..spNdim = u)
      const upsFlat = new Float64Array(ntpl * nar);
      if (Array.isArray(restartSp.ups[0]) || ArrayBuffer.isView(restartSp.ups[0])) {
        const rowLen = restartSp.ups[0].length;
        for (let r = 0; r < ntpl; r++) {
          const row = restartSp.ups[r];
          if (rowLen === nar) {
            for (let c = 0; c < nar; c++) upsFlat[r * nar + c] = row[c];
          } else {
            upsFlat[r * nar] = restartSp.t ? restartSp.t[r] : r / Math.max(1, ntpl - 1);
            for (let c = 0; c < spNdim; c++) upsFlat[r * nar + 1 + c] = row[c];
          }
        }
      } else if (restartSp.ups.length === ntpl * nar) {
        upsFlat.set(restartSp.ups);
      } else {
        for (let r = 0; r < ntpl; r++) {
          upsFlat[r * nar] = restartSp.t ? restartSp.t[r] : r / Math.max(1, ntpl - 1);
          for (let c = 0; c < spNdim; c++) upsFlat[r * nar + 1 + c] = restartSp.ups[r * spNdim + c];
        }
      }

      // Pack udotpsFlat as ntpl * spNdim
      const udotpsFlat = new Float64Array(ntpl * spNdim);
      if (restartSp.udotps && restartSp.udotps.length > 0) {
        if (Array.isArray(restartSp.udotps[0]) || ArrayBuffer.isView(restartSp.udotps[0])) {
          for (let r = 0; r < ntpl; r++) {
            const row = restartSp.udotps[r];
            for (let c = 0; c < spNdim; c++) udotpsFlat[r * spNdim + c] = row[c];
          }
        } else {
          udotpsFlat.set(restartSp.udotps);
        }
      }

      const spIcpPtr = mod._malloc(spNfpr * 4);
      const spRldotPtr = mod._malloc(spNfpr * 8);
      const upsPtr = mod._malloc(upsFlat.length * 8);
      const udotpsPtr = mod._malloc(udotpsFlat.length * 8);
      mod.HEAP32.set(spIcp, spIcpPtr >> 2);
      mod.HEAPF64.set(spRldot, spRldotPtr >> 3);
      mod.HEAPF64.set(upsFlat, upsPtr >> 3);
      mod.HEAPF64.set(udotpsFlat, udotpsPtr >> 3);

      mod._auto2000_set_restart_bvp_sp(
        restartSp.itp, restartSp.lab, nar, spNtst, spNcol,
        uPtr, restartSp.par.length, parPtr,
        spNfpr, spIcpPtr, spRldotPtr, upsPtr, udotpsPtr,
      );

      mod._free(spIcpPtr);
      mod._free(spRldotPtr);
      mod._free(upsPtr);
      mod._free(udotpsPtr);
    } else {
      mod._auto2000_set_restart_sp(restartSp.itp, restartSp.lab, ndim, uPtr, restartSp.par.length, parPtr, icpPtr);
    }

    mod._free(uPtr);
    mod._free(parPtr);
  }
  mod._free(icpPtr);
  if (iuzPtr) mod._free(iuzPtr);
  if (vuzPtr) mod._free(vuzPtr);

  const prevExitCode = typeof process !== 'undefined' ? process.exitCode : undefined;
  let rc = 0;
  let error = null;
  const t0 = performance.now();
  try {
    rc = mod._auto2000_run();
  } catch (err) {
    rc = -1;
    error = cbErr ?? err;
  } finally {
    if (typeof process !== 'undefined') process.exitCode = prevExitCode;
  }
  const runMs = performance.now() - t0;

  if (error) {
    return { rc, error, points: [], specialPoints: [], instantiateMs, runMs };
  }

  const numU = mod._auto2000_num_u();
  const isLC = ips !== 1 && ips !== 0;
  const points = [];
  for (let k = 0; k < numU; k++) {
    const parVals = Array.from({ length: activeIcp.length }, (_, i) => mod._auto2000_par(k, i));
    const ev = [];
    for (let j = 0; j < ndim; j++) {
      const r = mod._auto2000_ev_r(k, j);
      const im = mod._auto2000_ev_i(k, j);
      if (Number.isFinite(r) && Number.isFinite(im)) {
        ev.push({ r, i: im, mag: Math.hypot(r, im) });
      }
    }
    const uVals = isLC
      ? Array.from({ length: ndim }, (_, j) => mod._auto2000_usm(0, k, j))
      : Array.from({ length: ndim }, (_, j) => mod._auto2000_u(k, j));
    const uMin = isLC
      ? Array.from({ length: ndim }, (_, j) => mod._auto2000_usm(1, k, j))
      : undefined;
    points.push({
      par: parVals,
      u: uVals,
      ...(uMin !== undefined ? { uMin } : {}),
      ev,
      stable: classifyStability(ips, ev, effectiveIsw),
    });
  }

  const numSp = mod._auto2000_num_sp();
  const specialPoints = [];
  for (let i = 0; i < numSp; i++) {
    const nar = mod._auto2000_sp_nar(i);
    const nparx = mod._auto2000_sp_nparx(i);
    const itp = mod._auto2000_sp_itp(i);
    const spObj = {
      itp,
      ty: itpToType(itp),
      lab: mod._auto2000_sp_lab(i),
      par1: mod._auto2000_sp_par(i, 0),
      par2: mod._auto2000_sp_par(i, 1),
      par: Array.from({ length: nparx }, (_, p) => mod._auto2000_sp_par(i, p)),
      u1Based: Array.from({ length: nar }, (_, j) => mod._auto2000_sp_u(i, j)),
    };
    if (mod._auto2000_sp_has_mesh(i)) {
      const spNtst = mod._auto2000_sp_ntst(i);
      const spNcol = mod._auto2000_sp_ncol(i);
      const ntpl = mod._auto2000_sp_ntpl(i);
      const nfpr = mod._auto2000_sp_nfpr(i);
      const spNdim = nar - 1;
      spObj.ntst = spNtst;
      spObj.ncol = spNcol;
      spObj.ntpl = ntpl;
      spObj.nar = nar;
      spObj.nfpr = nfpr;
      spObj.icp = Array.from({ length: nfpr }, (_, p) => mod._auto2000_sp_icp(i, p));
      spObj.rldot = Array.from({ length: nfpr }, (_, p) => mod._auto2000_sp_rldot(i, p));
      spObj.t = Array.from({ length: ntpl }, (_, r) => mod._auto2000_sp_t(i, r));
      spObj.ups = Array.from({ length: ntpl }, (_, r) =>
        Array.from({ length: spNdim }, (_, c) => mod._auto2000_sp_ups(i, r, c + 1)),
      );
      spObj.udotps = Array.from({ length: ntpl }, (_, r) =>
        Array.from({ length: spNdim }, (_, c) => mod._auto2000_sp_udotps(i, r, c)),
      );
    }
    specialPoints.push(spObj);
  }

  // Codim-2 labeling on 2D equilibrium curves (LP2 / HB2)
  if ((ips === 1 || ips === 0) && Math.abs(effectiveIsw) === 2 && restartSp) {
    detectCodim2SpecialPoints({ mod, func, ndim, points, specialPoints, restartSp });
  }

  return { rc, numU, points, specialPoints, instantiateMs, runMs };
}
