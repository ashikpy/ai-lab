"use client";

import { BsLayoutThreeColumns } from "react-icons/bs";
import { AiFillHome } from "react-icons/ai";
import React, { useState } from "react";

function MobView({ className }) {
  const questions = Array.from({ length: 23 }, (_, i) => i + 1);

  return (
    <div
      className={`sidebar grid grid-cols-3 gap-2 p-5 ${className ? className : ""}`}
    >
      {questions.map((q, index) => (
        <a
          className="grid items-center rounded-md border-[1.5px] border-dashed border-[#ffffff24] p-2 text-center text-xl font-bold transition duration-200 hover:scale-105 hover:bg-[#ffffff24] hover:text-white"
          key={index}
          href={`/question/${q}`}
        >
          {q}
        </a>
      ))}
    </div>
  );
}

function Navbar() {
  const size = 20;

  const [showMobView, setShowMobView] = useState(false);

  return (
    <>
      {<MobView className={showMobView ? "active" : ""} />}
      <div
        className={`overlay ${showMobView ? "active" : ""} `}
        onClick={() => {
          setShowMobView(false);
        }}
      ></div>
      <nav className="text-bold p-4 lg:p-8">
        <div className="container flex items-center justify-between px-2 text-white">
          <BsLayoutThreeColumns
            size={size}
            onClick={() => {
              setShowMobView(!showMobView);
            }}
            className="cursor-pointer"
          />
          <h1 className="text-lg font-bold sm:text-xl lg:text-2xl">
            Data Structure Study Material
          </h1>
          <a href="/">
            <AiFillHome size={size} />
          </a>
        </div>
      </nav>
    </>
  );
}

export default Navbar;
