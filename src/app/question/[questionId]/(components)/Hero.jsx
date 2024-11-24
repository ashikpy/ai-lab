"use client";
import { FcPrevious } from "react-icons/fc";
import { FcNext } from "react-icons/fc";
import { MdOutlineContentPaste } from "react-icons/md";
import { CodeData } from "@/app/data";
import SyntaxHighlighter from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import { useState, useEffect } from "react";
import Link from "next/link";
import { BiLinkExternal } from "react-icons/bi";
import ReactMarkdown from "react-markdown";

function Hero({ questionId }) {

  const formattedTitle = CodeData[questionId].question
    .split(";")
    .map((line, index) => line.trim())
    .filter((line) => line)
    .map((line, index) => <p key={index}>{line}</p>);

  const [copy, setCopy] = useState(false);
  const [ifFirstPages, setIfFirstPages] = useState(false);
  const [ifLastPages, setIfLastPages] = useState(false);

  useEffect(() => {
    if (Number(questionId) === 1) {
      setIfFirstPages(true);
    } else {
      setIfFirstPages(false);
    }
    if (Number(questionId) === 23) {
      setIfLastPages(true);
    } else {
      setIfLastPages(false);
    }
  }, [questionId]);

  const copyCode = () => {
    navigator.clipboard.writeText(CodeData[questionId].code);
    setCopy(true);
    setTimeout(() => {
      setCopy(false);
    }, 2000);
  };

  return (
    <div className="container mt-5 flex flex-col gap-9 p-2 xl:flex-row xl:justify-between">
      <div className="flex flex-col gap-9 xl:max-w-[45vw]">
        <span className="flex flex-col gap-2">
          <h1 id={questionId} className="text-2xl font-black xl:text-3xl">
            Question - {questionId}
          </h1>
          <p>{formattedTitle}</p>
        </span>
        <span className="flex flex-col gap-2">
          {/* <h3 className="text-xl font-bold xl:text-2xl">Algorithm</h3> */}
          <ReactMarkdown>{CodeData[questionId].description}</ReactMarkdown>
        </span>
        <span className="hidden gap-5 xl:flex">
          <Link
            href={`/question/${Number(questionId) - 1} `}
            className={`flex w-max items-center gap-2 rounded-md bg-[#ededed] p-2 px-4 font-bold text-black ${ifFirstPages ? "cursor-not-allowed text-slate-400 line-through" : ""}`}
          >
            <FcPrevious />
            Question {Number(questionId) - 1}
          </Link>
          <Link
            href={`/`}
            className={`flex w-max items-center gap-2 rounded-md bg-[#ededed] p-2 px-4 font-bold text-black`}
          >
            Home
            <BiLinkExternal />
          </Link>
          <Link
            href={`/question/${Number(questionId) + 1}`}
            className={`flex w-max items-center gap-2 rounded-md bg-[#ededed] p-2 px-4 font-bold text-black ${ifLastPages ? "cursor-not-allowed text-slate-400 line-through" : ""}`}
          >
            Question {Number(questionId) + 1} <FcNext />
          </Link>
        </span>
      </div>

      <span className="flex flex-col gap-5">
        <h3 className="text-xl font-bold xl:text-2xl">Code</h3>
        <div className="flex items-center justify-between text-sm text-white">
          <Link
            href={questionId}
            className="rounded-md bg-[#ededed] p-2 font-bold text-black"
          >
            Question - {questionId}
          </Link>
          {copy ? (
            <button
              onClick={copyCode}
              className="flex items-center gap-2 rounded-md bg-black p-2 px-4 font-bold text-[#ededed]"
            >
              <MdOutlineContentPaste />
              Copied
            </button>
          ) : (
            <button
              onClick={copyCode}
              className="flex items-center gap-2 rounded-md bg-white p-2 font-bold text-black"
            >
              <MdOutlineContentPaste />
              Copy Code
            </button>
          )}
        </div>
        <SyntaxHighlighter
          language="python"
          style={atomOneDark}
          customStyle={{
            padding: "25px",
            borderRadius: "5px",
          }}
          wrapLongLines={true}
        >
          {CodeData[questionId].code}
        </SyntaxHighlighter>
        <span className="flex justify-around xl:hidden">
          <Link
            href={`/question/${Number(questionId) - 1} `}
            className={`flex w-max items-center gap-2 rounded-md bg-[#ededed] p-2 px-4 font-bold text-black ${ifFirstPages ? "cursor-not-allowed text-slate-400 line-through" : ""}`}
          >
            <FcPrevious />
            Question {Number(questionId) - 1}
          </Link>
          {/* <Link
            href={`/`}
            className={`flex w-max items-center gap-2 rounded-md bg-[#ededed] p-2 px-4 font-bold text-black`}
          >
            Home
            <BiLinkExternal />
          </Link> */}
          <Link
            href={`/question/${Number(questionId) + 1}`}
            className={`flex w-max items-center gap-2 rounded-md bg-[#ededed] p-2 px-4 font-bold text-black ${ifLastPages ? "cursor-not-allowed text-slate-400 line-through" : ""}`}
          >
            Question {Number(questionId) + 1} <FcNext />
          </Link>
        </span>
      </span>
    </div>
  );
}

export default Hero;
